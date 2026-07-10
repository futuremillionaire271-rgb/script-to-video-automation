"""
Animated "karaoke" captions — the strongest visual signal of a humanized,
modern YouTube/documentary edit.

Words appear as a bold lower-third block. The word being spoken is
highlighted with a colored pill and scaled up; already-spoken words stay
solid white; upcoming words are dimmed. Timing is distributed across the
scene duration (and, with a voiceover, the scene duration already matches
the narration), so the highlight tracks the voice.

Implementation notes:
- The caption only changes when the active word changes, so we render one
  PIL image per active-word state (~1 per word) and show each for its slice.
  That keeps rendering cheap even at hundreds of scenes.
- Layout is computed once at the base font size and never reflows; the
  active-word emphasis is drawn as a pill + color, not a size change to the
  layout, so text never jitters.
"""

import re
from dataclasses import dataclass

import numpy as np
from moviepy import CompositeVideoClip, ImageClip, VideoClip
from PIL import Image, ImageDraw, ImageFont

CAPTION_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Colors (RGBA)
COLOR_SPOKEN = (255, 255, 255, 255)      # words already said
COLOR_UPCOMING = (235, 235, 235, 130)    # words not yet said (dimmed)
COLOR_ACTIVE = (17, 17, 17, 255)         # active word text (dark, on pill)
PILL_COLOR = (255, 209, 71, 255)         # gold highlight pill
STROKE_COLOR = (0, 0, 0, 235)
SHADOW_COLOR = (0, 0, 0, 150)


@dataclass
class _Word:
    text: str        # display text (with original punctuation)
    x: float         # left x in the caption image
    y: float         # top y
    w: float         # rendered width
    is_key: bool     # part of the scene's keywords -> always emphasized


def _keyword_set(keywords) -> set[str]:
    words = set()
    for kw in keywords:
        for w in kw.lower().split():
            words.add(re.sub(r"[^\w'-]", "", w))
    return words


def _layout(text: str, font: ImageFont.FreeTypeFont, max_width: float,
            keywords: set[str], line_h: int, pad: int):
    """Wrap words to width; return (list[_Word], image_width, image_height)."""
    space_w = font.getlength(" ")
    tokens = text.split()

    # First pass: group into lines
    lines: list[list[str]] = [[]]
    widths = [0.0]
    for tok in tokens:
        tw = font.getlength(tok)
        cur = lines[-1]
        cur_w = widths[-1]
        if cur and cur_w + space_w + tw > max_width:
            lines.append([tok])
            widths.append(tw)
        else:
            widths[-1] = cur_w + (space_w if cur else 0) + tw
            cur.append(tok)

    img_w = int(max_width) + pad * 2
    img_h = line_h * len(lines) + pad * 2

    words: list[_Word] = []
    for li, line in enumerate(lines):
        line_w = widths[li]
        x = (img_w - line_w) / 2
        y = pad + li * line_h
        for tok in line:
            tw = font.getlength(tok)
            clean = re.sub(r"[^\w'-]", "", tok).lower()
            words.append(_Word(tok, x, y, tw, clean in keywords))
            x += tw + space_w
    return words, img_w, img_h


def _render_state(words, active_idx, img_w, img_h, font, line_h) -> np.ndarray:
    """Render the caption image for the given active-word index (RGBA array)."""
    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    stroke = max(3, font.size // 9)
    pill_pad_x = int(font.size * 0.22)
    pill_pad_y = int(font.size * 0.12)
    radius = int(font.size * 0.28)
    ascent, descent = font.getmetrics()
    text_h = ascent + descent

    for i, wd in enumerate(words):
        active = i == active_idx
        emphasized = active or wd.is_key

        # Drop shadow for all words (depth + legibility on any footage)
        draw.text((wd.x + stroke, wd.y + stroke), wd.text, font=font,
                  fill=SHADOW_COLOR, stroke_width=stroke, stroke_fill=SHADOW_COLOR)

        if active:
            # Gold pill behind the spoken word
            x0 = wd.x - pill_pad_x
            y0 = wd.y - pill_pad_y
            x1 = wd.x + wd.w + pill_pad_x
            y1 = wd.y + text_h + pill_pad_y
            draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=PILL_COLOR)
            draw.text((wd.x, wd.y), wd.text, font=font, fill=COLOR_ACTIVE)
        else:
            if i < active_idx:
                color = (255, 214, 92, 255) if wd.is_key else COLOR_SPOKEN
            else:
                color = (255, 214, 92, 190) if wd.is_key else COLOR_UPCOMING
            draw.text((wd.x, wd.y), wd.text, font=font, fill=color,
                      stroke_width=stroke, stroke_fill=STROKE_COLOR)

    return np.array(img)


def make_karaoke_caption(text: str, keywords, duration: float,
                         size: tuple[int, int]) -> VideoClip:
    """
    Build an animated karaoke caption clip (transparent background) for one
    scene. Composite it over the scene's video with .with_position.
    """
    frame_w, frame_h = size
    font_size = int(frame_h * 0.055)          # big, bold, readable
    font = ImageFont.truetype(CAPTION_FONT, font_size)
    max_width = int(frame_w * 0.80)
    line_h = int(font_size * 1.42)
    pad = int(font_size * 0.7)
    keyset = _keyword_set(list(keywords))

    words, img_w, img_h = _layout(text, font, max_width, keyset, line_h, pad)
    if not words:
        # No text: return a fully transparent clip
        empty = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        return ImageClip(empty).with_mask(
            ImageClip(np.zeros((frame_h, frame_w)), is_mask=True)
        ).with_duration(duration)

    # Distribute time across words, weighted by length (longer words dwell
    # a little longer) — approximates natural speech cadence.
    weights = np.array([max(1.0, len(w.text)) for w in words])
    ends = np.cumsum(weights) / weights.sum() * duration
    starts = np.concatenate([[0.0], ends[:-1]])

    # Position of the caption block: lower third, centered
    pos_x = (frame_w - img_w) // 2
    pos_y = int(frame_h - img_h - frame_h * 0.06)

    state_clips = []
    for i in range(len(words)):
        arr = _render_state(words, i, img_w, img_h, font, line_h)
        seg_dur = max(0.02, ends[i] - starts[i])
        rgb = arr[..., :3]
        alpha = arr[..., 3] / 255.0
        clip = (
            ImageClip(rgb)
            .with_mask(ImageClip(alpha, is_mask=True))
            .with_start(starts[i])
            .with_duration(seg_dur)
            .with_position((pos_x, pos_y))
        )
        state_clips.append(clip)

    return CompositeVideoClip(state_clips, size=size).with_duration(duration)


def burn_captions(clip: VideoClip, text: str, keywords) -> VideoClip:
    """Composite an animated karaoke caption over a scene clip."""
    caption = make_karaoke_caption(text, keywords, clip.duration, clip.size)
    return CompositeVideoClip([clip, caption], size=clip.size)
