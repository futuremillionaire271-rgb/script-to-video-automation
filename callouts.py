"""
On-screen entity callouts — a separate animated tag (independent of the
caption) for the "specific" things worth flagging: place names, person
names, and notable numbers / times / dates.

e.g. "...outdoors in Arizona..."          -> tag: ARIZONA
     "...twelve to sixteen ounces..."     -> tag: TWELVE TO SIXTEEN OUNCES
     "...go to bed at ten thirty..."       -> tag: TEN THIRTY

These add an "edited" editorial layer: the viewer's eye gets a crisp fact
card at the moment the narration names something concrete.
"""

import re

import numpy as np
from moviepy import CompositeVideoClip, ImageClip, VideoClip, vfx
from PIL import Image, ImageDraw, ImageFont

CALLOUT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
ACCENT = (255, 199, 51, 255)     # gold accent bar
BG = (18, 18, 22, 235)           # dark tag background
TEXT = (255, 255, 255, 255)

_NUMWORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand",
}
_CONNECT = {"to", "or", "and"}
_BIGNUM = {"ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
           "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
           "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
           "hundred", "thousand"}
_UNITS = {"ounces", "ounce", "hours", "hour", "minutes", "minute", "times",
          "glasses", "glass", "weeks", "week", "days", "day"}
# Common non-name capitalized sentence openers to ignore as "proper nouns"
_STOPCAPS = {"The", "They", "This", "That", "There", "Those", "Then", "These",
            "Your", "You", "When", "If", "For", "Do", "Now", "Start", "After",
            "Once", "But", "It", "In", "A", "An", "Move", "Think", "Drink",
            "Water", "Keep", "Try", "Continue", "Second", "First", "Suppose",
            "During", "Sometimes", "Most", "Every", "Pain", "Coffee",
            "Caffeine", "Many", "Sleep", "Dry", "Pouring", "Large", "Small",
            "Not", "Fluid", "Wake", "Produce"}


def extract_callout(text: str) -> str | None:
    """Pick the single best callout for a scene, or None."""
    # 1) Proper nouns: capitalized words that are not sentence-initial and
    #    not common capitalized openers.
    tokens = text.split()
    for i, tok in enumerate(tokens):
        word = re.sub(r"[^\w]", "", tok)
        if not word or not word[0].isupper() or not word.isalpha():
            continue
        prev = tokens[i - 1] if i > 0 else ""
        sentence_start = i == 0 or prev[-1:] in ".!?\""
        if sentence_start or word in _STOPCAPS or len(word) < 3:
            continue
        return word.upper()

    # 2) Number / time / date phrases.
    words = [re.sub(r"[^\w'-]", "", w).lower() for w in text.split()]
    n = len(words)
    best = None
    i = 0
    while i < n:
        if words[i] in _NUMWORDS:
            j = i
            span = [words[i]]
            k = i + 1
            while k < n and (words[k] in _NUMWORDS or
                             (words[k] in _CONNECT and k + 1 < n and words[k + 1] in _NUMWORDS)):
                span.append(words[k])
                k += 1
            num_count = sum(1 for w in span if w in _NUMWORDS)
            has_unit = k < n and words[k] in _UNITS
            has_big = any(w in _BIGNUM for w in span)  # >= ten, or a time/amount
            if has_unit:
                span.append(words[k])
            # Keep only substantive numbers: a unit (ounces/hours/times...) or
            # a "big" number (>=10, e.g. "ten thirty"). Bare small lists like
            # "two three" are skipped — they make poor callouts.
            if (num_count >= 2 or (num_count >= 1 and has_unit)) and (has_unit or has_big):
                if best is None or len(span) > len(best):
                    best = span
            i = k
        else:
            i += 1
    if best:
        return " ".join(best).upper()
    return None


def make_callout_clip(label: str, duration: float, size: tuple[int, int]) -> VideoClip:
    """A dark gold-accented tag in the upper-left, sliding + fading in/out."""
    frame_w, frame_h = size
    font_size = int(frame_h * 0.040)
    font = ImageFont.truetype(CALLOUT_FONT, font_size)

    text_w = int(font.getlength(label))
    ascent, descent = font.getmetrics()
    pad_x = int(font_size * 0.6)
    pad_y = int(font_size * 0.45)
    bar_w = int(font_size * 0.28)
    tag_w = bar_w + pad_x * 2 + text_w
    tag_h = ascent + descent + pad_y * 2

    img = Image.new("RGBA", (tag_w + 8, tag_h + 8), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    radius = int(tag_h * 0.18)
    draw.rounded_rectangle([0, 0, tag_w, tag_h], radius=radius, fill=BG)
    draw.rectangle([0, 0, bar_w, tag_h], fill=ACCENT)
    draw.text((bar_w + pad_x, pad_y), label, font=font, fill=TEXT)

    arr = np.array(img)
    x0, y0 = int(frame_w * 0.06), int(frame_h * 0.12)
    slide = 0.35

    def pos(t):
        k = min(t / slide, 1.0)
        ease = 1 - (1 - k) * (1 - k)
        return (x0 - int(40 * (1 - ease)), y0)

    hold = max(0.4, min(duration, 3.2))
    clip = (
        ImageClip(arr[..., :3])
        .with_mask(ImageClip(arr[..., 3] / 255.0, is_mask=True))
        .with_duration(hold)
        .with_position(pos)
        .with_effects([vfx.CrossFadeIn(0.3), vfx.CrossFadeOut(0.3)])
    )
    return clip


def add_callout(clip: VideoClip, text: str) -> VideoClip:
    """Composite an entity callout over a scene, if the text has one."""
    label = extract_callout(text)
    if not label:
        return clip
    tag = make_callout_clip(label, clip.duration, clip.size)
    return CompositeVideoClip([clip, tag], size=clip.size)
