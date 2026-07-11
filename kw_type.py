"""
Typewriter keyword overlay — a key term that types itself out on screen,
character by character, in a monospace "terminal" style with a blinking
cursor. Paired with per-character keyboard-click sounds (see sfx.py), it's
the recognizable pattern-interrupt used across faceless/tech YouTube edits.

One term per scene (when the scene has a memorable one), so the video gets
a typed keyword roughly every scene (~6-8s) as the user asked.
"""

import numpy as np
from moviepy import CompositeVideoClip, ImageClip, VideoClip
from PIL import Image, ImageDraw, ImageFont

MONO_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
GOLD = (255, 209, 71, 255)
CHIP_BG = (14, 16, 22, 225)
CHAR_TIME = 0.075          # seconds per typed character
HOLD = 1.6                 # seconds to hold after fully typed


def type_schedule(term: str, scene_start_rel: float) -> list[float]:
    """Absolute (scene-relative) times each character is committed — drives
    the keyboard-click SFX so clicks land exactly on each letter."""
    return [scene_start_rel + i * CHAR_TIME for i in range(len(term))]


def _render(term_upper: str, n_shown: int, cursor_on: bool, font, size):
    """Render the chip with the first n_shown chars + optional cursor."""
    frame_w, frame_h = size
    pad_x = int(font.size * 0.55)
    pad_y = int(font.size * 0.34)
    bar_w = int(font.size * 0.22)
    shown = term_upper[:n_shown]
    caret = "_" if cursor_on else " "
    display = shown + caret
    # width from the full term so the chip doesn't grow while typing
    full_w = int(font.getlength(term_upper + "_"))
    ascent, descent = font.getmetrics()
    chip_w = bar_w + pad_x * 2 + full_w
    chip_h = ascent + descent + pad_y * 2

    img = Image.new("RGBA", (chip_w + 8, chip_h + 8), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = int(chip_h * 0.16)
    d.rounded_rectangle([0, 0, chip_w, chip_h], radius=r, fill=CHIP_BG)
    d.rectangle([0, 0, bar_w, chip_h], fill=GOLD)
    d.text((bar_w + pad_x, pad_y), display, font=font, fill=GOLD)
    return np.array(img)


def make_typed_keyword(term: str, duration: float, size: tuple[int, int],
                       start_rel: float = 0.4) -> VideoClip:
    """
    Build the typing overlay for one scene. Positioned upper-center, above
    where the bottom caption sits. `start_rel` is when typing begins,
    relative to the scene start.
    """
    frame_w, frame_h = size
    term_upper = term.upper()
    font_size = int(frame_h * 0.046)
    font = ImageFont.truetype(MONO_FONT, font_size)

    n = len(term_upper)
    type_dur = n * CHAR_TIME
    # Precompute the distinct visual states: each char step (cursor on),
    # then a couple of blink frames during hold.
    states = []  # (image, start, dur)
    t = start_rel
    for i in range(1, n + 1):
        states.append((_render(term_upper, i, True, font, size), t, CHAR_TIME))
        t += CHAR_TIME
    # Hold with a blinking cursor (0.5s on/off)
    hold_end = min(duration, t + HOLD)
    blink = 0.5
    on = True
    while t < hold_end - 1e-3:
        seg = min(blink, hold_end - t)
        states.append((_render(term_upper, n, on, font, size), t, seg))
        on = not on
        t += seg

    # Position: upper third, left-aligned near the safe margin
    chip_h = states[0][0].shape[0]
    pos = (int(frame_w * 0.07), int(frame_h * 0.16))

    clips = []
    for arr, s, dur in states:
        clips.append(
            ImageClip(arr[..., :3])
            .with_mask(ImageClip(arr[..., 3] / 255.0, is_mask=True))
            .with_start(s).with_duration(max(0.03, dur)).with_position(pos)
        )
    return CompositeVideoClip(clips, size=size).with_duration(duration)


def add_typed_keyword(clip: VideoClip, term: str, start_rel: float = 0.4) -> VideoClip:
    if not term:
        return clip
    overlay = make_typed_keyword(term, clip.duration, clip.size, start_rel)
    return CompositeVideoClip([clip, overlay], size=clip.size)
