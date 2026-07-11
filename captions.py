"""
Animated "karaoke" captions, phrase-chunked.

Modern YouTube/documentary captions show a few words at a time — never a
whole paragraph — with the spoken word highlighted. This module:

- Splits a scene's text into short CHUNKS that each fit in at most 2 lines,
  breaking preferentially at punctuation (where a speaker pauses).
- Shows one chunk at a time for its slice of the scene, so no more than two
  lines are ever on screen.
- Within the active chunk, highlights the current word with a gold pill and
  keeps the scene's keywords gold. Because chunks are short and punctuation-
  aligned, the highlight tracks the voice far more tightly than a full
  sentence would.
"""

import re

import numpy as np
from moviepy import CompositeVideoClip, ImageClip, VideoClip
from PIL import Image, ImageDraw, ImageFont

CAPTION_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

COLOR_SPOKEN = (255, 255, 255, 255)
COLOR_UPCOMING = (232, 232, 232, 140)
COLOR_ACTIVE = (17, 17, 17, 255)
COLOR_KEY = (255, 214, 92, 255)
PILL_COLOR = (255, 209, 71, 255)
STROKE_COLOR = (0, 0, 0, 235)
SHADOW_COLOR = (0, 0, 0, 150)

MAX_LINES = 2
MAX_WORDS_PER_CHUNK = 7   # keeps highlight drift small within a chunk


_KW_FILLER = {"the", "and", "with", "from", "for", "into", "over", "out",
              "his", "her", "their", "a", "an", "of", "at", "in", "on", "to",
              "up", "by", "or", "man", "woman", "person", "people", "close",
              "view", "aerial", "home", "table", "background", "simple"}


def _keyword_set(keywords) -> set[str]:
    """
    Words to render gold in captions. Keywords may be phrase queries
    ('man drinking glass of water at night') — only their substantive
    words should glow, not fillers or generic staging words.
    """
    words = set()
    for kw in keywords:
        for w in kw.lower().split():
            w = re.sub(r"[^\w'-]", "", w)
            if len(w) >= 4 and w not in _KW_FILLER:
                words.add(w)
    return words


def _wrap_lines(words, font, max_width):
    """Return list of lines (each a list of words) for the given words."""
    space_w = font.getlength(" ")
    lines = [[]]
    width = 0.0
    for tok in words:
        tw = font.getlength(tok)
        if lines[-1] and width + space_w + tw > max_width:
            lines.append([tok])
            width = tw
        else:
            width += (space_w if lines[-1] else 0) + tw
            lines[-1].append(tok)
    return lines


def _chunk_text(text, font, max_width):
    """
    Split text into caption chunks that each fit in <= MAX_LINES lines.
    Prefer to end a chunk at punctuation so chunk boundaries fall on natural
    speech pauses.
    """
    tokens = text.split()
    chunks = []
    cur = []
    for tok in tokens:
        cur.append(tok)
        lines = _wrap_lines(cur, font, max_width)
        too_tall = len(lines) > MAX_LINES
        too_long = len(cur) >= MAX_WORDS_PER_CHUNK
        ends_sentence = tok[-1] in ".!?" if tok else False
        ends_clause = tok[-1] in ",;:" if tok else False

        if too_tall:
            # Roll the last word into a fresh chunk
            chunks.append(cur[:-1])
            cur = [tok]
        elif (ends_sentence and len(cur) >= 2) or (too_long and (ends_clause or ends_sentence)):
            chunks.append(cur)
            cur = []
        elif too_long:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c]


def _render_chunk_state(words, active_idx, keyset, img_w, img_h, font, line_h, pad):
    """Render one chunk with `active_idx` highlighted -> RGBA array."""
    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    stroke = max(3, font.size // 9)
    pill_px = int(font.size * 0.22)
    pill_py = int(font.size * 0.10)
    radius = int(font.size * 0.28)
    ascent, descent = font.getmetrics()
    text_h = ascent + descent
    space_w = font.getlength(" ")

    lines = _wrap_lines(words, font, img_w - pad * 2)
    idx = 0
    for li, line in enumerate(lines):
        line_w = sum(font.getlength(w) for w in line) + space_w * (len(line) - 1)
        x = (img_w - line_w) / 2
        y = pad + li * line_h
        for tok in line:
            clean = re.sub(r"[^\w'-]", "", tok).lower()
            is_key = clean in keyset
            active = idx == active_idx
            draw.text((x + stroke, y + stroke), tok, font=font, fill=SHADOW_COLOR,
                      stroke_width=stroke, stroke_fill=SHADOW_COLOR)
            if active:
                draw.rounded_rectangle(
                    [x - pill_px, y - pill_py, x + font.getlength(tok) + pill_px,
                     y + text_h + pill_py], radius=radius, fill=PILL_COLOR)
                draw.text((x, y), tok, font=font, fill=COLOR_ACTIVE)
            else:
                if idx < active_idx:
                    color = COLOR_KEY if is_key else COLOR_SPOKEN
                else:
                    color = (255, 214, 92, 190) if is_key else COLOR_UPCOMING
                draw.text((x, y), tok, font=font, fill=color,
                          stroke_width=stroke, stroke_fill=STROKE_COLOR)
            x += font.getlength(tok) + space_w
            idx += 1
    return np.array(img)


def _word_windows(chunks, duration: float, word_times):
    """
    Absolute display windows for every (chunk, word) state.

    With word_times (real Whisper timestamps per token, relative to the
    scene start): the highlight moves exactly when the narrator says each
    word — a chunk appears at its first word and holds until the next
    chunk's first word.

    Without word_times: estimated windows weighted by word length.
    Returns [(chunk_idx, word_idx, start, end)].
    """
    windows = []
    if word_times is not None:
        tok = 0
        chunk_spans = []
        for words in chunks:
            chunk_spans.append((tok, tok + len(words)))
            tok += len(words)
        for ci, (a, b) in enumerate(chunk_spans):
            chunk_disp_end = (chunk_spans[ci + 1][0] < len(word_times)
                              and word_times[chunk_spans[ci + 1][0]][0]
                              if ci + 1 < len(chunk_spans) else duration) or duration
            for wi, ti in enumerate(range(a, b)):
                w_start = word_times[ti][0] if wi > 0 else (
                    word_times[a][0] if ci > 0 else 0.0)
                w_end = (word_times[ti + 1][0] if ti + 1 < b else chunk_disp_end)
                windows.append((ci, wi, w_start, max(w_end, w_start + 0.03)))
    else:
        chunk_weights = np.array([sum(len(w) for w in c) for c in chunks], dtype=float)
        chunk_ends = np.cumsum(chunk_weights) / chunk_weights.sum() * duration
        chunk_starts = np.concatenate([[0.0], chunk_ends[:-1]])
        for ci, words in enumerate(chunks):
            c_start = chunk_starts[ci]
            c_dur = max(0.05, chunk_ends[ci] - c_start)
            w_weights = np.array([max(1.0, len(w)) for w in words], dtype=float)
            w_ends = np.cumsum(w_weights) / w_weights.sum() * c_dur
            w_starts = np.concatenate([[0.0], w_ends[:-1]])
            for wi in range(len(words)):
                windows.append((ci, wi, c_start + w_starts[wi],
                                c_start + w_ends[wi]))
    return windows


def make_karaoke_caption(text: str, keywords, duration: float,
                         size: tuple[int, int], word_times=None) -> VideoClip:
    """
    Animated, phrase-chunked karaoke caption (transparent background).
    word_times: optional [(start, end)] per word of text.split(), relative
    to the scene start (from Whisper alignment) — locks the highlight to
    the narrator's actual voice.
    """
    frame_w, frame_h = size
    font_size = int(frame_h * 0.052)
    font = ImageFont.truetype(CAPTION_FONT, font_size)
    max_width = int(frame_w * 0.78)
    line_h = int(font_size * 1.4)
    pad = int(font_size * 0.6)
    keyset = _keyword_set(list(keywords))

    chunks = _chunk_text(text, font, max_width)
    if not chunks:
        empty = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        return ImageClip(empty).with_mask(
            ImageClip(np.zeros((frame_h, frame_w)), is_mask=True)
        ).with_duration(duration)

    if word_times is not None and len(word_times) != sum(len(c) for c in chunks):
        word_times = None  # token mismatch — fall back to estimates

    img_h = line_h * MAX_LINES + pad * 2
    pos_y = int(frame_h - img_h - frame_h * 0.055)
    state_clips = []

    for ci, wi, w_start, w_end in _word_windows(chunks, duration, word_times):
        w_start = min(max(w_start, 0.0), max(duration - 0.03, 0.0))
        w_end = min(max(w_end, w_start + 0.03), duration)
        arr = _render_chunk_state(chunks[ci], wi, keyset, frame_w, img_h,
                                  font, line_h, pad)
        clip = (
            ImageClip(arr[..., :3])
            .with_mask(ImageClip(arr[..., 3] / 255.0, is_mask=True))
            .with_start(w_start)
            .with_duration(w_end - w_start)
            .with_position((0, pos_y))
        )
        state_clips.append(clip)

    return CompositeVideoClip(state_clips, size=size).with_duration(duration)


def burn_captions(clip: VideoClip, text: str, keywords, word_times=None) -> VideoClip:
    caption = make_karaoke_caption(text, keywords, clip.duration, clip.size,
                                   word_times=word_times)
    return CompositeVideoClip([clip, caption], size=clip.size)
