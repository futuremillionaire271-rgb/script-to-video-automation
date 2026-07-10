"""
Documentary-style visual effects: per-scene motion (Ken Burns), punch-in,
color grade, vignette, and a varied transition plan across scene boundaries.

All effects are baked into each scene file at render time, so the final
ffmpeg concat stays a lossless stream copy.
"""

import random

import numpy as np
from moviepy import CompositeVideoClip, ImageClip, VideoClip, vfx

# Transition styles at a scene boundary. Hard cuts dominate (as in real
# documentary editing); dips and punches are accents.
_BOUNDARY_STYLES = ["cut"] * 9 + ["black"] * 4 + ["white"] * 3 + ["punch"] * 4
_MOTION_STYLES = ["zoom-in", "zoom-out", "pan-left", "pan-right"]

FADE_BLACK_DURATION = 0.16
FADE_WHITE_DURATION = 0.12


def plan_transitions(n_scenes: int, seed: int = 42) -> list[str]:
    """
    Assign a transition style to each of the n-1 scene boundaries.
    Deterministic (seeded) so interrupted runs resume with the same plan.
    Never repeats the same non-cut style on consecutive boundaries.
    """
    rng = random.Random(seed)
    styles: list[str] = []
    prev = None
    for _ in range(max(0, n_scenes - 1)):
        pool = [s for s in _BOUNDARY_STYLES if s == "cut" or s != prev]
        style = rng.choice(pool)
        styles.append(style)
        prev = style
    return styles


def plan_motion(n_scenes: int, seed: int = 7) -> list[str]:
    """
    Assign a Ken Burns motion style per scene, deterministic, never the
    same style twice in a row.
    """
    rng = random.Random(seed)
    styles: list[str] = []
    prev = None
    for _ in range(n_scenes):
        pool = [s for s in _MOTION_STYLES if s != prev]
        style = rng.choice(pool)
        styles.append(style)
        prev = style
    return styles


def apply_motion(clip: VideoClip, style: str, strength: float = 0.07) -> VideoClip:
    """
    Ken Burns motion baked into a fitted (frame-sized) clip: slow zoom in,
    zoom out, or lateral pan. Subtle by design — it should read as "alive",
    not as an effect.
    """
    w, h = clip.size
    d = max(clip.duration, 0.01)

    if style == "zoom-in":
        moving = clip.resized(lambda t: 1.0 + strength * (t / d)).with_position("center")
    elif style == "zoom-out":
        moving = clip.resized(lambda t: (1.0 + strength) - strength * (t / d)).with_position("center")
    else:
        scale = 1.0 + strength
        big = clip.resized(scale)
        dx = big.w - w
        dy = (big.h - h) / 2
        if style == "pan-left":
            pos = lambda t: (-dx * (t / d), -dy)
        else:  # pan-right
            pos = lambda t: (-dx * (1.0 - t / d), -dy)
        moving = big.with_position(pos)

    return CompositeVideoClip([moving], size=(w, h)).with_duration(clip.duration)


def punch_in(clip: VideoClip, amount: float = 0.10, settle: float = 0.3) -> VideoClip:
    """
    Punch-in opener: the scene starts slightly zoomed and snaps to rest in
    the first `settle` seconds. Used as the in-side of a 'punch' boundary.
    """
    w, h = clip.size

    def scale(t):
        if t >= settle:
            return 1.0
        k = 1.0 - t / settle
        return 1.0 + amount * k * k  # ease-out

    return CompositeVideoClip(
        [clip.resized(scale).with_position("center")], size=(w, h)
    ).with_duration(clip.duration)


def apply_grade(clip: VideoClip) -> VideoClip:
    """Subtle contrast lift — a light 'graded' look."""
    return clip.with_effects([vfx.LumContrast(lum=0, contrast=0.10, contrast_threshold=127)])


_VIGNETTE_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def add_vignette(clip: VideoClip, strength: float = 0.32) -> VideoClip:
    """Darken frame corners slightly to focus the eye center-frame."""
    w, h = clip.size
    if (w, h) not in _VIGNETTE_CACHE:
        y, x = np.ogrid[:h, :w]
        r = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2)
        alpha = np.clip((r - 0.72) / 0.75, 0.0, 1.0) * strength
        _VIGNETTE_CACHE[(w, h)] = (np.zeros((h, w, 3), dtype=np.uint8), alpha)

    black, alpha = _VIGNETTE_CACHE[(w, h)]
    overlay = (
        ImageClip(black)
        .with_mask(ImageClip(alpha, is_mask=True))
        .with_duration(clip.duration)
    )
    return CompositeVideoClip([clip, overlay], size=(w, h))


def scene_edge_styles(index: int, n_scenes: int, boundaries: list[str]) -> tuple[str, str]:
    """
    Resolve the in/out treatment for scene `index` from the boundary plan.
    First scene fades in from black; last scene fades out to black.
    A 'punch' boundary is a hard cut out of the previous scene and a
    punch-in opener on the next.
    """
    in_style = "black" if index == 0 else boundaries[index - 1]
    out_style = "black" if index == n_scenes - 1 else boundaries[index]
    if out_style == "punch":
        out_style = "cut"  # punch lives on the incoming side
    return in_style, out_style


def apply_edges(clip: VideoClip, in_style: str, out_style: str) -> VideoClip:
    """Bake the boundary treatments into the clip's first/last frames."""
    effects = []
    if in_style == "black":
        effects.append(vfx.FadeIn(FADE_BLACK_DURATION))
    elif in_style == "white":
        effects.append(vfx.FadeIn(FADE_WHITE_DURATION, initial_color=[255, 255, 255]))

    if out_style == "black":
        effects.append(vfx.FadeOut(FADE_BLACK_DURATION))
    elif out_style == "white":
        effects.append(vfx.FadeOut(FADE_WHITE_DURATION, final_color=[255, 255, 255]))

    return clip.with_effects(effects) if effects else clip
