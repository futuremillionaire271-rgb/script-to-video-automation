"""
Stock footage search, download, and trim — built for 450+ scene runs.

Step 3: Search Pexels and Pixabay for clips matching scene keywords,
        alternating provider per scene to spread rate-limit load.
        Results are cached for 24h; requests are throttled to stay inside
        each provider's limits (Pexels 200/hr + 20k/month, Pixabay 100/60s).
Step 4: Download the chosen clip, trim/fit to the scene, and delete the raw
        download immediately so hundreds of raw files never accumulate.

Requires PEXELS_API_KEY and PIXABAY_API_KEY in .env.

`make_placeholder_clip` remains the no-network --demo fallback.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from moviepy import CompositeVideoClip, ImageClip, TextClip, VideoClip, VideoFileClip, vfx

from api_limits import MonthlyBudget, RateLimiter, SearchCache
from scene_parser import Scene

# override=True so .env is the source of truth even when the shell already
# has (possibly stale) PEXELS_API_KEY / PIXABAY_API_KEY values set
load_dotenv(override=True)

VIDEO_SIZE = (1920, 1080)
REQUEST_TIMEOUT = 20
RESULTS_PER_QUERY = 30  # more results per request = fewer requests + more variety
VISION_TOP_K = 6            # strong queries mean few candidates need a look
VISION_ACCEPT = 0.24        # strong visual match: take it and stop searching
VISION_MIN = 0.19           # after all queries, never pick below this silently

# Throttles sized just under the documented provider limits
_PEXELS_LIMITER = RateLimiter("pexels", 190, 3600)      # limit: 200/hour
_PIXABAY_LIMITER = RateLimiter("pixabay", 90, 60)       # limit: 100/60s
_PEXELS_BUDGET = MonthlyBudget("pexels", 19_500)        # limit: 20,000/month
_CACHE = SearchCache()


class ClipSearchError(RuntimeError):
    """Raised when stock footage cannot be sourced — never fail silently."""


@dataclass
class ClipCandidate:
    """A chosen stock video file, ready to download."""
    source: str          # "pexels" or "pixabay"
    video_id: str
    download_url: str
    width: int
    height: int
    duration: float      # source video duration in seconds
    query: str
    sim: float = -1.0    # CLIP visual match score (-1 = not scored)
    desc: str = ""       # slug/tags (human-readable identity)
    thumb: str = ""      # first thumbnail (for audit)

    @property
    def key(self) -> str:
        return f"{self.source}:{self.video_id}"


# ---------------------------------------------------------------------------
# Step 3: search (throttled + cached)
# ---------------------------------------------------------------------------

def check_api_access() -> None:
    """
    Preflight before any scene work: verify keys are loaded and at least one
    stock API is reachable. Raises ClipSearchError with the exact reason.
    (Goes through the same cache/throttle path as real searches, so a warm
    cache makes this free on resumed runs.)
    """
    for name in ("PEXELS_API_KEY", "PIXABAY_API_KEY"):
        print(f"  {name}: "
              f"{'present (' + str(len(os.environ[name])) + ' chars)' if os.getenv(name) else 'MISSING'}")
    print(f"  Pexels monthly usage: {_PEXELS_BUDGET.used()}/{_PEXELS_BUDGET.limit}")

    failures = []
    for provider in ("pexels", "pixabay"):
        try:
            _search(provider, "nature")
            print(f"  {provider} reachability: OK")
        except requests.RequestException as exc:
            failures.append(f"{provider} unreachable: {exc}")
        except RuntimeError as exc:
            failures.append(f"{provider}: {exc}")

    if len(failures) == 2:
        raise ClipSearchError(
            "No stock footage API is usable:\n  - " + "\n  - ".join(failures)
            + "\nIf errors mention 'Tunnel connection failed: 403', this environment's "
            "network policy is blocking the stock footage domains — allow "
            "api.pexels.com, *.pexels.com, pixabay.com, cdn.pixabay.com, or run locally."
        )
    if failures:
        print(f"  WARNING: {failures[0]} — continuing with the other source")


def _search(provider: str, query: str) -> list[dict]:
    """Cached, throttled search. Returns a list of raw candidate dicts."""
    cache_key = f"{provider}:{query}"
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    if provider == "pexels":
        results = _pexels_request(query)
    else:
        results = _pixabay_request(query)
    _CACHE.put(cache_key, results)
    return results


def _pexels_request(query: str) -> list[dict]:
    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        return []
    _PEXELS_LIMITER.wait()
    _PEXELS_BUDGET.increment()
    resp = requests.get(
        "https://api.pexels.com/videos/search",
        headers={"Authorization": api_key},
        params={"query": query, "orientation": "landscape", "per_page": RESULTS_PER_QUERY},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()

    results = []
    for video in resp.json().get("videos", []):
        file = _pick_pexels_file(video.get("video_files", []))
        if file is None:
            continue
        results.append({
            "source": "pexels",
            "id": str(video["id"]),
            "url": file["link"],
            "width": file["width"],
            "height": file["height"],
            "duration": float(video.get("duration", 0)),
            # Pexels page URL carries a descriptive slug, e.g.
            # ".../video/a-woman-drinking-water-12345/" -> relevance signal
            "desc": video.get("url", ""),
            "thumbs": _pexels_thumbs(video),
        })
    return results


def _pexels_thumbs(video: dict) -> list[str]:
    """Up to 3 frames spread across the video — one frame can lie."""
    thumbs = [video.get("image", "")]
    pics = video.get("video_pictures", [])
    if pics:
        for idx in {len(pics) // 2, len(pics) - 1}:
            url = pics[idx].get("picture", "")
            if url:
                thumbs.append(url)
    return [t for t in thumbs if t][:3]


def _pick_pexels_file(video_files: list[dict]) -> dict | None:
    """Pick the best rendition of a Pexels video: mp4, landscape, near-1080p."""
    candidates = [
        f for f in video_files
        if f.get("file_type") == "video/mp4"
        and f.get("width") and f.get("height")
        and f["width"] > f["height"]
    ]
    if not candidates:
        return None
    # Closest width to 1920 wins; avoids tiny previews and huge 4K downloads
    return min(candidates, key=lambda f: abs(f["width"] - VIDEO_SIZE[0]))


def _pixabay_request(query: str) -> list[dict]:
    api_key = os.getenv("PIXABAY_API_KEY")
    if not api_key:
        return []
    _PIXABAY_LIMITER.wait()
    resp = requests.get(
        "https://pixabay.com/api/videos/",
        params={"key": api_key, "q": query, "per_page": RESULTS_PER_QUERY},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()

    results = []
    for hit in resp.json().get("hits", []):
        renditions = hit.get("videos", {})
        file = renditions.get("large") or renditions.get("medium")
        if not file or not file.get("url"):
            continue
        width, height = file.get("width", 0), file.get("height", 0)
        if width <= height:
            continue
        results.append({
            "source": "pixabay",
            "id": str(hit["id"]),
            "url": file["url"],
            "width": width,
            "height": height,
            "duration": float(hit.get("duration", 0)),
            # Pixabay provides comma-separated tags -> relevance signal
            "desc": hit.get("tags", ""),
            "thumbs": [file.get("thumbnail", "")],
        })
    return results


def _score(width: int, height: int, duration: float, needed: float) -> float:
    """Rank a candidate on technical quality: long enough > HD > near-1080p."""
    score = 0.0
    if duration >= needed:
        score += 6.0
    else:
        score -= 4.0 * (needed - duration)  # penalize clips that must loop
    if width >= 1920:
        score += 2.0
    elif width >= 1280:
        score += 1.0
    score -= abs(width - VIDEO_SIZE[0]) / 1920.0  # prefer near-1080p files
    return score


_PERSON_WORDS = {"man", "woman", "person", "people", "adult", "senior",
                 "guy", "lady", "male", "female", "human", "patient"}
_ANIMAL_WORDS = {"dog", "cat", "pet", "puppy", "kitten", "bird", "animal",
                 "wildlife", "horse", "cow", "duck", "deer", "insect", "bee",
                 "wasp", "hornet", "squirrel", "fox", "monkey", "lamb", "sheep",
                 "goat", "calf", "chicken", "rooster", "hen", "pig", "piglet",
                 "rabbit", "bunny", "fish", "kitten", "kitty", "pony", "donkey",
                 "elephant", "lion", "tiger", "bear", "wolf", "turtle", "frog"}
_DESC_SPLIT = re.compile(r"[^a-z]+")
_FILLER = {"the", "and", "with", "from", "for", "into", "his", "her", "their",
           "who", "that", "this", "then", "them", "out", "off", "over"}


def _relevance_floor(query: str) -> float:
    """
    Minimum relevance a candidate must reach to be picked normally.
    Multi-word queries must share at least one meaningful word with the
    clip's description; single-word anchors just can't be person/animal
    mismatched (negative).
    """
    content_words = [w for w in query.lower().split()
                     if len(w) > 2 and w not in _FILLER]
    return 3.0 if len(content_words) >= 2 else 0.0


def _relevance(query: str, desc: str) -> float:
    """
    How well a candidate's description (Pexels URL slug / Pixabay tags)
    matches the search query. This is the difference between "man drinking
    water" returning a person vs. a dog at a bowl.
    """
    if not desc:
        return 0.0
    q_words = {w for w in query.lower().split() if len(w) > 2}
    d_words = {w for w in _DESC_SPLIT.split(desc.lower()) if len(w) > 2}
    if not q_words or not d_words:
        return 0.0

    overlap = q_words & d_words
    score = 3.0 * len(overlap)

    # If we asked for a person but the clip is clearly an animal (and not
    # also a person), demote it hard.
    if (q_words & _PERSON_WORDS) and (d_words & _ANIMAL_WORDS) \
            and not (d_words & _PERSON_WORDS):
        score -= 8.0

    return score


def find_clip_for_scene(scene: Scene, index: int, used_ids: set[str],
                        use_vision: bool = True) -> ClipCandidate | None:
    """
    Find the best stock clip for a scene — VISION-GATED.

    Text metadata only pre-filters; CLIP looks at every surviving
    candidate's actual frames and the highest visual match wins. A strong
    match (>= VISION_ACCEPT) is taken immediately; otherwise every query is
    tried and the best seen wins if it clears VISION_MIN. Anything weaker
    is a loudly-logged last resort, never a silent pick.
    """
    if any(" " in kw for kw in scene.keywords):
        queries = list(dict.fromkeys(kw for kw in scene.keywords if kw))
    else:
        queries = list(dict.fromkeys(q for q in [
            " ".join(scene.keywords),
            " ".join(scene.keywords[:2]),
            scene.keywords[0] if scene.keywords else "",
        ] if q))
    if not queries:
        return None

    providers = ("pexels", "pixabay") if index % 2 == 0 else ("pixabay", "pexels")
    vision_on = use_vision
    if vision_on:
        import vision
        vision_on = vision.available()

    errors: list[str] = []
    attempts = 0
    best_global: ClipCandidate | None = None
    best_used, best_used_score = None, float("-inf")
    best_weak, best_weak_score = None, float("-inf")

    for query in queries:
        for provider in providers:
            attempts += 1
            try:
                results = _search(provider, query)
            except requests.RequestException as exc:
                errors.append(f"{provider}('{query}'): {exc}")
                continue

            floor = _relevance_floor(query)
            passing = []
            for r in results:
                rel = _relevance(query, r.get("desc", ""))
                score = (_score(r["width"], r["height"], r["duration"], scene.duration)
                         + rel)
                thumbs = tuple(r.get("thumbs") or ([r["thumb"]] if r.get("thumb") else []))
                candidate = ClipCandidate(
                    source=r["source"], video_id=r["id"], download_url=r["url"],
                    width=r["width"], height=r["height"], duration=r["duration"],
                    query=query, desc=r.get("desc", ""),
                    thumb=thumbs[0] if thumbs else "",
                )
                if candidate.key in used_ids:
                    if rel >= floor and score > best_used_score:
                        best_used, best_used_score = candidate, score
                elif rel < floor:
                    if score > best_weak_score:
                        best_weak, best_weak_score = candidate, score
                else:
                    passing.append((score, candidate, thumbs))

            if not passing:
                continue
            passing.sort(key=lambda t: t[0], reverse=True)
            top = passing[:VISION_TOP_K]

            if vision_on:
                import vision
                sims = vision.score_candidates(query, [t[2] for t in top])
                for (score, cand, _), sim in zip(top, sims):
                    if sim is None:
                        continue
                    cand.sim = sim
                    if best_global is None or sim > best_global.sim:
                        best_global = cand
                # Strong visual match for THIS query -> done
                if best_global is not None and best_global.sim >= VISION_ACCEPT \
                        and best_global.query == query:
                    used_ids.add(best_global.key)
                    return best_global
            else:
                # No vision available: best text-ranked candidate wins
                best = top[0][1]
                used_ids.add(best.key)
                return best

    if best_global is not None and best_global.sim >= VISION_MIN:
        used_ids.add(best_global.key)
        return best_global
    if best_global is not None:
        print(f"  Scene {index + 1}: LOW VISUAL MATCH (sim={best_global.sim:.2f}) "
              f"for {queries} — best available used; consider editing this shot")
        used_ids.add(best_global.key)
        return best_global
    if best_used is not None:
        print(f"  Scene {index + 1}: all candidates already used — reusing "
              f"{best_used.key} for '{best_used.query}'")
        return best_used
    if best_weak is not None:
        print(f"  Scene {index + 1}: WARNING — only low-relevance results for "
              f"{queries}; using {best_weak.key} (consider editing this shot)")
        used_ids.add(best_weak.key)
        return best_weak

    if errors and len(errors) == attempts:
        raise ClipSearchError(
            f"All {attempts} search attempts failed for keywords {scene.keywords}:\n  - "
            + "\n  - ".join(errors)
        )
    return None


# ---------------------------------------------------------------------------
# Step 4: download & trim (raw file deleted by the caller after render)
# ---------------------------------------------------------------------------

def download_clip(candidate: ClipCandidate, dest: Path, retries: int = 4) -> Path:
    """
    Download a stock clip to dest (streamed to keep memory flat).

    Retries on transient network errors (connection reset, timeout) with
    exponential backoff — a single blip must never crash a long render.
    """
    import time as _time
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_exc = None
    for attempt in range(retries):
        try:
            with requests.get(candidate.download_url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            if dest.stat().st_size > 0:
                return dest
            raise requests.RequestException("empty download")
        except (requests.RequestException, OSError) as exc:
            last_exc = exc
            dest.unlink(missing_ok=True)
            if attempt < retries - 1:
                _time.sleep(2 ** attempt)  # 1s, 2s, 4s
    raise ClipSearchError(
        f"download failed after {retries} attempts for {candidate.key}: {last_exc}"
    )


def fetch_scene_clip(scene: Scene, index: int, raw_dir: Path, used_ids: set[str],
                     duration: float = None, shot: int = 0) -> tuple[VideoClip, Path]:
    """
    Steps 3-4 for one shot of a scene: search, download, trim to the shot
    duration (defaults to the whole scene), fit to 1920x1080. Returns
    (clip, raw_path); the caller renders the scene file and then deletes
    raw_path so raw downloads never pile up.

    Scenes longer than the pacing target are built from multiple shots
    (shot=0,1,...) — each shot gets a different clip for the same keywords,
    keeping visuals moving without splitting the caption.

    Raises ClipSearchError if no stock clip can be sourced — placeholders
    are only ever used in explicit --demo mode, never as a silent fallback.
    """
    shot_duration = duration if duration is not None else scene.duration
    candidate = find_clip_for_scene(scene, index, used_ids)
    if candidate is None:
        raise ClipSearchError(
            f"Scene {index + 1}: no stock results on Pexels or Pixabay for "
            f"keywords {scene.keywords} (searches succeeded but returned nothing "
            f"usable — try broader keywords)"
        )

    raw_path = raw_dir / (
        f"raw_{index + 1:04d}_{shot}_{candidate.source}_{candidate.video_id}.mp4"
    )
    download_clip(candidate, raw_path)

    clip = VideoFileClip(str(raw_path)).without_audio()

    # Trim to the shot duration; loop if the source is shorter
    if clip.duration >= shot_duration:
        clip = clip.subclipped(0, shot_duration)
    else:
        clip = clip.with_effects([vfx.Loop(duration=shot_duration)])

    return _fit_to_frame(clip), raw_path


def _fit_to_frame(clip: VideoClip, size: tuple[int, int] = VIDEO_SIZE) -> VideoClip:
    """Scale to cover the target frame, then center-crop to exactly fit."""
    target_w, target_h = size
    if clip.w / clip.h >= target_w / target_h:
        clip = clip.resized(height=target_h)
    else:
        clip = clip.resized(width=target_w)
    return clip.cropped(
        width=target_w, height=target_h,
        x_center=clip.w / 2, y_center=clip.h / 2,
    )


# ---------------------------------------------------------------------------
# Demo fallback: placeholder clips (no network required)
# ---------------------------------------------------------------------------

# Scene background palette for placeholder clips: (top color, bottom color)
_GRADIENTS = [
    ((242, 155, 74), (94, 34, 98)),    # sunset orange -> dusk purple
    ((34, 87, 46), (10, 30, 18)),      # forest green -> deep shade
    ((52, 120, 168), (16, 42, 78)),    # ocean blue -> deep sea
    ((150, 170, 200), (240, 245, 250)),  # mountain slate -> snow white
    ((120, 90, 160), (30, 20, 50)),    # violet -> night
    ((200, 120, 80), (80, 40, 30)),    # clay -> umber
]

_PLACEHOLDER_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _gradient_image(size: tuple[int, int], top_rgb, bottom_rgb) -> np.ndarray:
    """Build a vertical gradient frame as a numpy array."""
    width, height = size
    top = np.array(top_rgb, dtype=np.float64)
    bottom = np.array(bottom_rgb, dtype=np.float64)
    t = np.linspace(0.0, 1.0, height)[:, None, None]
    rows = top[None, None, :] * (1 - t) + bottom[None, None, :] * t
    return np.broadcast_to(rows, (height, width, 3)).astype(np.uint8)


def make_placeholder_clip(scene: Scene, index: int, size: tuple[int, int] = VIDEO_SIZE) -> VideoClip:
    """
    Generate a placeholder clip for a scene: a gradient background labeled
    with the scene's search keywords. Stands in for a downloaded stock clip
    so the pipeline can be demoed without API keys or network access.
    """
    width, height = size
    top, bottom = _GRADIENTS[index % len(_GRADIENTS)]
    background = ImageClip(_gradient_image(size, top, bottom)).with_duration(scene.duration)

    keyword_label = TextClip(
        font=_PLACEHOLDER_FONT,
        text=" ".join(scene.keywords),
        font_size=int(height * 0.065),
        color="white",
        method="label",
    ).with_duration(scene.duration).with_position(("center", int(height * 0.38)))

    note = TextClip(
        font=_PLACEHOLDER_FONT,
        text=f"[ placeholder for stock clip — scene {index + 1} ]",
        font_size=int(height * 0.022),
        color="#cccccc",
        method="label",
    ).with_duration(scene.duration).with_position(("center", int(height * 0.50)))

    return CompositeVideoClip([background, keyword_label, note], size=size)
