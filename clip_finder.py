"""
Stock footage search, download, and trim.

Step 3: Search Pexels (primary) and Pixabay (fallback) for clips matching
        scene keywords. Prefer landscape, HD, duration >= scene duration.
Step 4: Download the chosen clip and trim/resize it to the scene's duration
        at 1920x1080.

Requires PEXELS_API_KEY and PIXABAY_API_KEY in .env.

`make_placeholder_clip` remains available as the no-network demo fallback
(used by --demo, and per-scene when no stock result is found).
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from moviepy import CompositeVideoClip, ImageClip, TextClip, VideoClip, VideoFileClip, vfx

from scene_parser import Scene

load_dotenv()

VIDEO_SIZE = (1920, 1080)
TEMP_DIR = Path("temp")
REQUEST_TIMEOUT = 20


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


# ---------------------------------------------------------------------------
# Step 3: search
# ---------------------------------------------------------------------------

def search_pexels(query: str, needed_duration: float) -> ClipCandidate | None:
    """Search Pexels videos; return the best candidate or None."""
    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        return None

    resp = requests.get(
        "https://api.pexels.com/videos/search",
        headers={"Authorization": api_key},
        params={"query": query, "orientation": "landscape", "per_page": 15},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()

    best, best_score = None, float("-inf")
    for video in resp.json().get("videos", []):
        file = _pick_pexels_file(video.get("video_files", []))
        if file is None:
            continue
        score = _score(file["width"], file["height"], video.get("duration", 0), needed_duration)
        if score > best_score:
            best_score = score
            best = ClipCandidate(
                source="pexels",
                video_id=str(video["id"]),
                download_url=file["link"],
                width=file["width"],
                height=file["height"],
                duration=float(video.get("duration", 0)),
                query=query,
            )
    return best


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


def search_pixabay(query: str, needed_duration: float) -> ClipCandidate | None:
    """Search Pixabay videos; return the best candidate or None."""
    api_key = os.getenv("PIXABAY_API_KEY")
    if not api_key:
        return None

    resp = requests.get(
        "https://pixabay.com/api/videos/",
        params={"key": api_key, "q": query, "per_page": 15},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()

    best, best_score = None, float("-inf")
    for hit in resp.json().get("hits", []):
        renditions = hit.get("videos", {})
        file = renditions.get("large") or renditions.get("medium")
        if not file or not file.get("url"):
            continue
        width, height = file.get("width", 0), file.get("height", 0)
        if width <= height:
            continue
        score = _score(width, height, hit.get("duration", 0), needed_duration)
        if score > best_score:
            best_score = score
            best = ClipCandidate(
                source="pixabay",
                video_id=str(hit["id"]),
                download_url=file["url"],
                width=width,
                height=height,
                duration=float(hit.get("duration", 0)),
                query=query,
            )
    return best


def _score(width: int, height: int, duration: float, needed: float) -> float:
    """Rank a candidate: long enough > HD > close to target resolution."""
    score = 0.0
    if duration >= needed:
        score += 10.0
    else:
        score -= 5.0 * (needed - duration)  # penalize clips that must loop
    if width >= 1920:
        score += 3.0
    elif width >= 1280:
        score += 1.5
    score -= abs(width - VIDEO_SIZE[0]) / 1920.0  # prefer near-1080p files
    return score


def find_clip_for_scene(scene: Scene, used_ids: set[str]) -> ClipCandidate | None:
    """
    Find the best stock clip for a scene, trying progressively broader
    queries: all keywords -> first two -> first one; Pexels first, then
    Pixabay. Skips clips already used in this run.
    """
    queries = [
        " ".join(scene.keywords),
        " ".join(scene.keywords[:2]),
        scene.keywords[0] if scene.keywords else "",
    ]
    # Deduplicate while preserving order, drop empties
    queries = list(dict.fromkeys(q for q in queries if q))

    for search in (search_pexels, search_pixabay):
        for query in queries:
            try:
                candidate = search(query, scene.duration)
            except requests.RequestException as exc:
                print(f"  ! {search.__name__}('{query}') failed: {exc}")
                continue
            if candidate and f"{candidate.source}:{candidate.video_id}" not in used_ids:
                used_ids.add(f"{candidate.source}:{candidate.video_id}")
                return candidate
    return None


# ---------------------------------------------------------------------------
# Step 4: download & trim
# ---------------------------------------------------------------------------

def download_clip(candidate: ClipCandidate, dest: Path) -> Path:
    """Download a stock clip to dest (streamed to keep memory flat)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(candidate.download_url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return dest


def prepare_scene_clip(scene: Scene, index: int, temp_dir: Path = TEMP_DIR) -> VideoClip:
    """
    Full Steps 3-4 for one scene: search, download, trim to scene duration,
    and fit to 1920x1080 (scale to cover, center-crop).

    Falls back to a placeholder clip if no stock result is found.
    """
    candidate = find_clip_for_scene(scene, prepare_scene_clip._used_ids)
    if candidate is None:
        print(f"  Scene {index + 1}: no stock match for {scene.keywords}, using placeholder")
        return make_placeholder_clip(scene, index)

    path = temp_dir / f"scene_{index + 1:02d}_{candidate.source}_{candidate.video_id}.mp4"
    print(f"  Scene {index + 1}: {candidate.source} #{candidate.video_id} "
          f"({candidate.width}x{candidate.height}, {candidate.duration:.0f}s) "
          f"for '{candidate.query}'")
    download_clip(candidate, path)

    clip = VideoFileClip(str(path)).without_audio()

    # Trim to scene duration; loop if the source is shorter
    if clip.duration >= scene.duration:
        clip = clip.subclipped(0, scene.duration)
    else:
        clip = clip.with_effects([vfx.Loop(duration=scene.duration)])

    return _fit_to_frame(clip)


# Track clip IDs used this run so scenes don't repeat footage
prepare_scene_clip._used_ids = set()


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
