"""
Visual verification of stock candidates with CLIP — the boss of selection.

"Don't trust the search engine — look at the clip." Every candidate's
thumbnail frames are scored against the scene's query by CLIP (open_clip
ViT-B-32). Text metadata only pre-filters; the PICTURE decides. Candidates
below a real match threshold are never picked silently.

- thumbnails fetched concurrently (8 threads), scored in one batch
- multiple frames per video are averaged (one frame can lie)
- scores cached to cache/vision_scores.json so re-runs are instant
- CPU-only; if the model is unavailable, callers fall back to text ranking
"""

import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

_STATE = None  # (model, preprocess, tokenizer, torch) or False if unavailable
_CACHE_PATH = Path("cache/vision_scores.json")
_cache: dict | None = None


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(_CACHE_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            _cache = {}
    return _cache


def _save_cache() -> None:
    if _cache is not None:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(_cache))
        tmp.replace(_CACHE_PATH)


def _key(query: str, urls: tuple[str, ...]) -> str:
    return hashlib.sha1(("|".join(urls) + "::" + query).encode()).hexdigest()[:20]


def _load():
    global _STATE
    if _STATE is None:
        try:
            import open_clip
            import torch
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            model.eval()
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            _STATE = (model, preprocess, tokenizer, torch)
            print("  vision: CLIP ViT-B-32 loaded (selection is vision-gated)")
        except Exception as exc:  # missing package, blocked download, ...
            print(f"  vision: unavailable ({str(exc)[:120]}) — text-only ranking")
            _STATE = False
    return _STATE


def available() -> bool:
    return bool(_load())


def _fetch(url: str):
    from PIL import Image
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def score_candidates(query: str, thumb_sets: list[tuple[str, ...]]) -> list[float | None]:
    """
    CLIP-score each candidate against the query. Each candidate may have
    several thumbnail frames — their similarities are averaged. Returns one
    similarity per candidate (None if every frame failed to fetch).
    """
    state = _load()
    if not state:
        return [None] * len(thumb_sets)
    model, preprocess, tokenizer, torch = state

    caption = f"a video of {query}"
    cache = _load_cache()
    out: list[float | None] = [None] * len(thumb_sets)
    to_fetch: list[tuple[int, str]] = []  # (candidate_idx, url)

    for i, urls in enumerate(thumb_sets):
        urls = tuple(u for u in urls if u)
        if not urls:
            continue
        k = _key(caption, urls)
        if k in cache:
            out[i] = cache[k]
        else:
            for u in urls[:3]:
                to_fetch.append((i, u))

    if to_fetch:
        with ThreadPoolExecutor(max_workers=8) as pool:
            images = list(pool.map(lambda t: _fetch(t[1]), to_fetch))

        batch, owners = [], []
        for (i, _), img in zip(to_fetch, images):
            if img is not None:
                batch.append(preprocess(img))
                owners.append(i)

        if batch:
            with torch.no_grad():
                feats = model.encode_image(torch.stack(batch))
                feats /= feats.norm(dim=-1, keepdim=True)
                text = model.encode_text(tokenizer([caption]))
                text /= text.norm(dim=-1, keepdim=True)
                sims = (feats @ text.T).squeeze(1).tolist()

            per_cand: dict[int, list[float]] = {}
            for i, s in zip(owners, sims):
                per_cand.setdefault(i, []).append(float(s))
            for i, vals in per_cand.items():
                out[i] = sum(vals) / len(vals)
                urls = tuple(u for u in thumb_sets[i] if u)
                cache[_key(caption, urls)] = round(out[i], 4)
            _save_cache()

    return out
