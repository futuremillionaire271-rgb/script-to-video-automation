"""
Documentary / modern-YouTube visual effects, tuned to be *visible*:
strong Ken Burns motion, punch-ins, a teal-orange cinematic grade, vignette,
and a varied transition plan across scene boundaries.

All effects are baked into each scene file at render time, so the final
ffmpeg concat stays a lossless stream copy.
"""

import random

import numpy as np
from moviepy import CompositeVideoClip, ImageClip, VideoClip, vfx

# Transition styles at a scene boundary. A mix that stays lively without
# feeling gimmicky: hard cuts as the backbone, plus punch-ins, dip-to-black,
# white flashes, and zoom-punches as accents.
_BOUNDARY_STYLES = ["cut"] * 6 + ["punch"] * 5 + ["black"] * 3 + ["white"] * 2 + ["zoompunch"] * 4
_MOTION_STYLES = ["zoom-in", "zoom-out", "pan-left", "pan-right", "pan-up"]

FADE_BLACK_DURATION = 0.22
FADE_WHITE_DURATION = 0.10

# Motion is deliberately strong now — the previous 7% over 7s was invisible.
MOTION_STRENGTH = 0.22


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
        styles.append(rng.choice(pool))
        prev = styles[-1]
    return styles


def plan_motion(n_scenes: int, seed: int = 7) -> list[str]:
    """Assign a Ken Burns motion style per scene, never the same twice running."""
    rng = random.Random(seed)
    styles: list[str] = []
    prev = None
    for _ in range(n_scenes):
        pool = [s for s in _MOTION_STYLES if s != prev]
        styles.append(rng.choice(pool))
        prev = styles[-1]
    return styles


def apply_motion(clip: VideoClip, style: str, strength: float = MOTION_STRENGTH) -> VideoClip:
    """
    Strong, continuous Ken Burns motion baked into a fitted (frame-sized)
    clip: zoom in/out or a lateral/vertical pan. Eased so it accelerates and
    settles instead of moving linearly.
    """
    w, h = clip.size
    d = max(clip.duration, 0.01)

    def ease(t):  # smoothstep 0->1
        x = min(max(t / d, 0.0), 1.0)
        return x * x * (3 - 2 * x)

    if style == "zoom-in":
        moving = clip.resized(lambda t: 1.0 + strength * ease(t)).with_position("center")
    elif style == "zoom-out":
        moving = clip.resized(lambda t: (1.0 + strength) - strength * ease(t)).with_position("center")
    else:
        # Pan across an over-scaled frame so edges never show
        scale = 1.0 + strength
        big = clip.resized(scale)
        dx = big.w - w
        dy = big.h - h
        if style == "pan-left":
            pos = lambda t: (-dx * ease(t), -dy / 2)
        elif style == "pan-right":
            pos = lambda t: (-dx * (1.0 - ease(t)), -dy / 2)
        elif style == "pan-up":
            pos = lambda t: (-dx / 2, -dy * ease(t))
        else:  # pan-down
            pos = lambda t: (-dx / 2, -dy * (1.0 - ease(t)))
        moving = big.with_position(pos)

    return CompositeVideoClip([moving], size=(w, h)).with_duration(clip.duration)


def punch_in(clip: VideoClip, amount: float = 0.32, settle: float = 0.28) -> VideoClip:
    """
    Snappy punch-in opener: the scene slams in slightly zoomed and eases to
    rest in `settle` seconds. Clearly visible — used as the incoming side of
    'punch' boundaries.
    """
    w, h = clip.size

    def scale(t):
        if t >= settle:
            return 1.0
        k = 1.0 - t / settle
        return 1.0 + amount * (k * k)  # ease-out

    return CompositeVideoClip(
        [clip.resized(scale).with_position("center")], size=(w, h)
    ).with_duration(clip.duration)


def zoom_punch_out(clip: VideoClip, amount: float = 0.14, dur: float = 0.25) -> VideoClip:
    """A quick zoom bump at the very end of a scene — a visible 'kick' out."""
    w, h = clip.size
    d = max(clip.duration, 0.01)
    start = d - dur

    def scale(t):
        if t <= start:
            return 1.0
        k = (t - start) / dur
        return 1.0 + amount * (k * k)

    return CompositeVideoClip(
        [clip.resized(scale).with_position("center")], size=(w, h)
    ).with_duration(clip.duration)


def apply_grade(clip: VideoClip, strength: float = 1.0) -> VideoClip:
    """
    Teal-orange cinematic grade: lift/cool the shadows toward teal, warm the
    highlights toward orange, add contrast and a little saturation. Vectorized
    per frame — clearly visible but tuned not to wreck the footage.
    """
    def grade(frame):
        f = frame.astype(np.float32) / 255.0

        # S-curve contrast around mid-gray
        c = 1.0 + 0.18 * strength
        f = np.clip((f - 0.5) * c + 0.5, 0.0, 1.0)

        # Luma to find shadows vs highlights
        luma = f @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        shadow = np.clip(1.0 - luma * 1.6, 0.0, 1.0)[..., None]
        highl = np.clip((luma - 0.5) * 2.0, 0.0, 1.0)[..., None]

        teal = np.array([-0.04, 0.02, 0.06], dtype=np.float32) * strength
        warm = np.array([0.07, 0.03, -0.05], dtype=np.float32) * strength
        f = f + shadow * teal + highl * warm

        # Saturation boost
        gray = (f @ np.array([0.299, 0.587, 0.114], dtype=np.float32))[..., None]
        f = gray + (f - gray) * (1.0 + 0.15 * strength)

        return (np.clip(f, 0.0, 1.0) * 255).astype("uint8")

    return clip.image_transform(grade)


_VIGNETTE_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def add_vignette(clip: VideoClip, strength: float = 0.40) -> VideoClip:
    """Darken frame corners to focus the eye center-frame (cinematic)."""
    w, h = clip.size
    if (w, h) not in _VIGNETTE_CACHE:
        y, x = np.ogrid[:h, :w]
        r = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2)
        alpha = np.clip((r - 0.68) / 0.75, 0.0, 1.0) * strength
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
    Resolve in/out treatment for scene `index` from the boundary plan.
    First scene fades in from black; last fades out to black. 'punch' and
    'zoompunch' live on the incoming side, so they leave the previous scene
    on a hard cut.
    """
    in_style = "black" if index == 0 else boundaries[index - 1]
    out_style = "black" if index == n_scenes - 1 else boundaries[index]
    if out_style in ("punch", "zoompunch"):
        out_style = "cut"
    return in_style, out_style


def apply_edges(clip: VideoClip, in_style: str, out_style: str) -> VideoClip:
    """Bake boundary treatments into the clip's first/last frames."""
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
