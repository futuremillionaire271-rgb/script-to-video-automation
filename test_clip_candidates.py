"""
Focused regression checks for preview candidate ranking.

Run with:
    python test_clip_candidates.py
"""

import sys
import types

import clip_finder as cf
from scene_parser import Scene


def test_ranked_candidates() -> None:
    original_search = cf._search
    original_vision = sys.modules.get("vision")

    try:
        def fake_search(_provider: str, query: str) -> list[dict]:
            return [
                {
                    "source": "pexels",
                    "id": "city-low",
                    "url": "https://example.com/low.mp4",
                    "width": 1920,
                    "height": 1080,
                    "duration": 5.0,
                    "desc": f"{query} skyline traffic",
                    "thumbs": ["https://example.com/low.jpg"],
                },
                {
                    "source": "pixabay",
                    "id": "city-high",
                    "url": "https://example.com/high.mp4",
                    "width": 1920,
                    "height": 1080,
                    "duration": 6.0,
                    "desc": f"{query} street lights",
                    "thumbs": ["https://example.com/high.jpg"],
                },
            ]

        cf._search = fake_search
        sys.modules["vision"] = types.SimpleNamespace(
            available=lambda: True,
            score_candidates=lambda _query, thumbs: [0.21, 0.34][: len(thumbs)],
        )

        scene = Scene("City traffic at night.", 0.0, 3.0, ["city night traffic"])
        results = cf.find_candidates_for_scene(scene, 0, set(), use_vision=True, limit=2)

        assert len(results) == 2, f"expected 2 candidates, got {len(results)}"
        assert results[0].video_id == "city-high", "highest vision score should rank first"
        assert results[0].sim == 0.34, "top candidate should carry the vision score"
    finally:
        cf._search = original_search
        if original_vision is None:
            sys.modules.pop("vision", None)
        else:
            sys.modules["vision"] = original_vision


def test_weak_fallback() -> None:
    original_search = cf._search
    try:
        def fake_search(_provider: str, _query: str) -> list[dict]:
            return [
                {
                    "source": "pexels",
                    "id": "fallback",
                    "url": "https://example.com/fallback.mp4",
                    "width": 1280,
                    "height": 720,
                    "duration": 4.0,
                    "desc": "abstract stock footage",
                    "thumbs": ["https://example.com/fallback.jpg"],
                }
            ]

        cf._search = fake_search
        scene = Scene("Pouring water in a glass.", 0.0, 3.0, ["pouring glass water"])
        results = cf.find_candidates_for_scene(scene, 0, set(), use_vision=False, limit=1)

        assert len(results) == 1, "weak fallback should still return one preview candidate"
        assert results[0].video_id == "fallback", "fallback candidate should be surfaced"
    finally:
        cf._search = original_search


if __name__ == "__main__":
    test_ranked_candidates()
    test_weak_fallback()
    print("test_clip_candidates.py: all checks passed")
