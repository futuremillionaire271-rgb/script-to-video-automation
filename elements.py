"""
Statement cards — full-screen typography moments.

Retention psychology: when the narration lands a short, powerful line
("The solution is not to stop drinking water."), the strongest edit is not
another B-roll clip — it's the LINE ITSELF, huge and bold, owning the whole
screen. Synced text forces the viewer to read along with the voice
(dual-coding), and the sudden format change is a pattern interrupt.

Cards are used sparingly (never two close together) so they keep their
punch.
"""

import re

import numpy as np
from moviepy import CompositeVideoClip, ImageClip, VideoClip, vfx
from PIL import Image, ImageDraw, ImageFont

CARD_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
GOLD = (255, 209, 71, 255)
WHITE = (245, 246, 248, 255)

# Lines that open with these read as thesis statements -> card-worthy
_CARD_CUES = (
    "the solution", "the basic strategy", "the important principle",
    "start with", "think of", "remember", "the better question",
    "there is another", "the goal is", "those are not always",
    "a self-reinforcing",
)
_MAX_CARD_WORDS = 10
_MIN_SCENES_BETWEEN_CARDS = 4


def plan_cards(scenes) -> list[bool]:
    """Decide which scenes become statement cards (sparse, deterministic)."""
    flags = [False] * len(scenes)
    last = -_MIN_SCENES_BETWEEN_CARDS
    for i, sc in enumerate(scenes):
        if i - last < _MIN_SCENES_BETWEEN_CARDS or i == 0:
            continue
        words = len(sc.text.split())
        if words > _MAX_CARD_WORDS:
            continue
        t = sc.text.lower()
        if any(t.startswith(c) for c in _CARD_CUES) or words <= 6:
            flags[i] = True
            last = i
    return flags


def _card_background(size: tuple[int, int]) -> np.ndarray:
    """Deep navy radial gradient — premium, calm, high text contrast."""
    w, h = size
    y, x = np.ogrid[:h, :w]
    r = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2)
    center = np.array([24, 30, 44], dtype=np.float64)
    edge = np.array([8, 10, 16], dtype=np.float64)
    t = np.clip(r / 1.35, 0, 1)[..., None]
    return (center * (1 - t) + edge * t).astype(np.uint8)


def make_statement_card(text: str, keywords, duration: float,
                        size: tuple[int, int]) -> VideoClip:
    """Full-screen bold statement with a slow push-in and pop entrance."""
    frame_w, frame_h = size
    font_size = int(frame_h * 0.082)
    font = ImageFont.truetype(CARD_FONT, font_size)
    max_width = int(frame_w * 0.78)

    keyset = set()
    for kw in keywords:
        for w in str(kw).lower().split():
            keyset.add(re.sub(r"[^\w'-]", "", w))

    # Wrap
    space_w = font.getlength(" ")
    lines, width = [[]], 0.0
    for tok in text.split():
        tw = font.getlength(tok)
        if lines[-1] and width + space_w + tw > max_width:
            lines.append([tok])
            width = tw
        else:
            width += (space_w if lines[-1] else 0) + tw
            lines[-1].append(tok)

    line_h = int(font_size * 1.35)
    block_h = line_h * len(lines)
    img = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    y = (frame_h - block_h) // 2
    accent_words = 0
    for line in lines:
        line_w = sum(font.getlength(w) for w in line) + space_w * (len(line) - 1)
        x = (frame_w - line_w) / 2
        for tok in line:
            clean = re.sub(r"[^\w'-]", "", tok).lower()
            emphasize = (clean in keyset or (len(clean) >= 5 and accent_words < 2
                                             and clean not in ("their", "there")))
            color = GOLD if emphasize and accent_words < 3 else WHITE
            if color == GOLD:
                accent_words += 1
            draw.text((x, y), tok, font=font, fill=color)
            x += font.getlength(tok) + space_w
        y += line_h

    # Gold rule under the block — a small designed touch
    rule_w = int(frame_w * 0.12)
    ry = (frame_h + block_h) // 2 + int(frame_h * 0.035)
    draw.rounded_rectangle(
        [(frame_w - rule_w) // 2, ry, (frame_w + rule_w) // 2, ry + int(frame_h * 0.008)],
        radius=4, fill=GOLD)

    arr = np.array(img)
    bg = ImageClip(_card_background(size)).with_duration(duration)
    text_clip = (
        ImageClip(arr[..., :3])
        .with_mask(ImageClip(arr[..., 3] / 255.0, is_mask=True))
        .with_duration(duration)
        .with_effects([vfx.CrossFadeIn(0.18)])
    )
    card = CompositeVideoClip([bg, text_clip], size=size).with_duration(duration)

    # Slow push-in keeps the card alive for its whole duration
    d = max(duration, 0.01)
    return CompositeVideoClip(
        [card.resized(lambda t: 1.0 + 0.05 * (t / d)).with_position("center")],
        size=size,
    ).with_duration(duration)
