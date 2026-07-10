"""
Stock footage search and download.

Step 3: Search Pexels/Pixabay APIs for clips matching scene keywords.
Step 4: Download and trim clips to match scene duration.

Until the APIs are wired up (needs PEXELS_API_KEY / PIXABAY_API_KEY in .env),
`make_placeholder_clip` generates a stand-in visual per scene so the rest of
the pipeline (assembly, captions, export) can run end-to-end in demo mode.
"""

import numpy as np
from moviepy import CompositeVideoClip, ImageClip, TextClip, VideoClip

from scene_parser import Scene

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


def make_placeholder_clip(scene: Scene, index: int, size: tuple[int, int] = (1920, 1080)) -> VideoClip:
    """
    Generate a placeholder clip for a scene: a gradient background labeled
    with the scene's search keywords. Stands in for a downloaded stock clip
    so the pipeline can be demoed without API keys.
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


# TODO Step 3: Pexels API search (https://api.pexels.com/videos/search)
#   - query = " ".join(scene.keywords), orientation=landscape
#   - prefer HD video_files, duration >= scene.duration
# TODO Step 3b: Pixabay fallback (https://pixabay.com/api/videos/)
# TODO Step 4: download chosen clip to temp/, trim/loop to scene.duration,
#   resize/crop to 1920x1080
