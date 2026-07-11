"""
Visual verification of stock candidates with CLIP.

"Don't trust the search engine — look at the clip." For each candidate we
fetch its thumbnail image and score it against the scene's query with a
CLIP model (open_clip ViT-B-32). A clip whose PICTURE doesn't match the
words is rejected no matter how well its text metadata scored.

CPU-only, thumbnail-only (no video downloads), ~100ms per image. The model
loads lazily once per process; if open_clip or the model weights are
unavailable, callers get None scores and fall back to text-only ranking.
"""

import io
import os

import requests

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

_STATE = None  # (model, preprocess, tokenizer, torch) or False if unavailable


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
            print("  vision: CLIP ViT-B-32 loaded")
        except Exception as exc:  # missing package, blocked download, ...
            print(f"  vision: unavailable ({str(exc)[:120]}) — text-only ranking")
            _STATE = False
    return _STATE


def available() -> bool:
    return bool(_load())


def score_candidates(query: str, thumb_urls: list[str]) -> list[float | None]:
    """
    CLIP-score each thumbnail against the query. Returns cosine similarities
    (typically 0.15 poor .. 0.35 strong), or None per-image on fetch errors.
    """
    state = _load()
    if not state:
        return [None] * len(thumb_urls)
    model, preprocess, tokenizer, torch = state

    from PIL import Image

    images, idx_map = [], []
    for i, url in enumerate(thumb_urls):
        if not url:
            continue
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            images.append(preprocess(img))
            idx_map.append(i)
        except Exception:
            continue

    scores: list[float | None] = [None] * len(thumb_urls)
    if not images:
        return scores

    with torch.no_grad():
        image_batch = torch.stack(images)
        image_feat = model.encode_image(image_batch)
        image_feat /= image_feat.norm(dim=-1, keepdim=True)
        text_feat = model.encode_text(tokenizer([query]))
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        sims = (image_feat @ text_feat.T).squeeze(1).tolist()

    for i, s in zip(idx_map, sims):
        scores[i] = float(s)
    return scores


def vision_bonus(sim: float | None) -> float:
    """
    Convert a CLIP similarity into a ranking bonus on the same scale as the
    text relevance score. 0.20 is roughly 'plausibly related'; below that
    counts against the candidate, above counts for it.
    """
    if sim is None:
        return 0.0
    return (sim - 0.20) * 40.0
