"""
Video editing and assembly (moviepy v2 API + ffmpeg concat demuxer).

Step 5: Assemble scene clips. The final stitch uses ffmpeg's concat demuxer
        with stream copy — near-instant and flat-memory at any scale.
        Every scene file is rendered with identical codec settings so
        lossless concat is valid.
Step 6: Burn in captions — custom PIL renderer so the scene's key words can
        be highlighted in gold inside the white caption text.
Step 7: Export final MP4, 1920x1080, H.264, then mux background music
        (ducked under narration) and/or a voiceover track without touching
        the video stream.
"""

import re
import subprocess
from pathlib import Path

import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe
from moviepy import CompositeVideoClip, ImageClip, VideoClip
from PIL import Image, ImageDraw, ImageFont

VIDEO_SIZE = (1920, 1080)
FPS = 24
CAPTION_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
HIGHLIGHT_COLOR = (255, 214, 66, 255)   # gold for the scene's key words
CAPTION_COLOR = (255, 255, 255, 255)
MUSIC_VOLUME = 0.10                     # background bed level under narration

# Uniform encoder settings for every scene file — REQUIRED for the concat
# demuxer's stream copy to produce a valid output
_SCENE_ENCODE_ARGS = ["-crf", "23", "-pix_fmt", "yuv420p"]


# ---------------------------------------------------------------------------
# Step 6: captions with keyword highlighting
# ---------------------------------------------------------------------------

def _keyword_word_set(keywords: list[str]) -> set[str]:
    """Individual lowercase words from the scene's keywords."""
    words = set()
    for kw in keywords:
        for w in kw.lower().split():
            words.add(w)
    return words


def add_caption(clip: VideoClip, text: str, keywords: list[str] = ()) -> VideoClip:
    """
    Burn caption text onto the bottom of a clip (Step 6). Words that appear
    in the scene's keywords are rendered in gold; the rest in white, all
    with a black stroke for readability on any footage.
    """
    frame_w, frame_h = clip.size
    font_size = int(frame_h * 0.044)
    font = ImageFont.truetype(CAPTION_FONT, font_size)
    max_width = int(frame_w * 0.82)
    highlight = _keyword_word_set(list(keywords))

    # Wrap words to caption width
    space_w = font.getlength(" ")
    lines: list[list[str]] = []
    current: list[str] = []
    current_w = 0.0
    for word in text.split():
        ww = font.getlength(word)
        if current and current_w + space_w + ww > max_width:
            lines.append(current)
            current, current_w = [word], ww
        else:
            current_w += (space_w if current else 0) + ww
            current.append(word)
    if current:
        lines.append(current)

    line_h = int(font_size * 1.32)
    pad = font_size // 2
    img_h = line_h * len(lines) + pad * 2
    img = Image.new("RGBA", (frame_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    stroke = max(2, font_size // 12)

    for li, line in enumerate(lines):
        total_w = sum(font.getlength(w) for w in line) + space_w * (len(line) - 1)
        x = (frame_w - total_w) / 2
        y = pad + li * line_h
        for word in line:
            clean = re.sub(r"[^\w'-]", "", word).lower()
            color = HIGHLIGHT_COLOR if clean in highlight else CAPTION_COLOR
            draw.text((x, y), word, font=font, fill=color,
                      stroke_width=stroke, stroke_fill=(0, 0, 0, 230))
            x += font.getlength(word) + space_w

    arr = np.array(img)
    caption = (
        ImageClip(arr[..., :3])
        .with_mask(ImageClip(arr[..., 3] / 255.0, is_mask=True))
        .with_duration(clip.duration)
        .with_position(("center", frame_h - img_h - int(frame_h * 0.055)))
    )
    return CompositeVideoClip([clip, caption], size=clip.size)


# ---------------------------------------------------------------------------
# Per-scene render + fast final assembly
# ---------------------------------------------------------------------------

def render_scene_file(clip: VideoClip, path: Path) -> Path:
    """
    Encode one finished scene (trimmed, captioned, transitioned) to disk
    with the uniform settings that make lossless final concat possible.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    clip.write_videofile(
        str(path),
        fps=FPS,
        codec="libx264",
        audio=False,
        preset="veryfast",
        threads=4,
        ffmpeg_params=_SCENE_ENCODE_ARGS,
        logger=None,  # silence per-frame progress bars (hundreds of renders)
    )
    return path


def concat_scene_files(scene_files: list[Path], output_path: Path) -> Path:
    """
    Final assembly (Step 5) via ffmpeg's concat demuxer with stream copy:
    no re-encode, near-instant, flat memory — unlike moviepy, which gets
    slow and memory-heavy with hundreds of clips.
    """
    if not scene_files:
        raise ValueError("No scene files to concatenate")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_file = output_path.with_suffix(".concat.txt")
    list_file.write_text(
        "".join(f"file '{Path(f).resolve()}'\n" for f in scene_files)
    )

    cmd = [
        get_ffmpeg_exe(), "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    list_file.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg concat failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
        )
    return output_path


# ---------------------------------------------------------------------------
# Step 7b: audio — narration + ducked background music
# ---------------------------------------------------------------------------

def mux_audio(video_path: Path, duration: float,
              music: Path = None, voiceover: Path = None,
              music_volume: float = MUSIC_VOLUME) -> Path:
    """
    Add audio to the assembled video without re-encoding the video stream.

    - voiceover plays at full volume (the star of the mix)
    - music is looped to cover the video, ducked to `music_volume`, with a
      fade-in and a fade-out at the end, so it supports the narration
      instead of fighting it
    """
    if music is None and voiceover is None:
        return video_path

    tmp_out = video_path.with_name(video_path.stem + ".audio.mp4")
    cmd = [get_ffmpeg_exe(), "-y", "-i", str(video_path)]

    music_idx = voice_idx = None
    next_idx = 1
    if music is not None:
        cmd += ["-stream_loop", "-1", "-i", str(music)]
        music_idx, next_idx = next_idx, next_idx + 1
    if voiceover is not None:
        cmd += ["-i", str(voiceover)]
        voice_idx = next_idx

    filters = []
    if music_idx is not None:
        fade_out_start = max(0.0, duration - 2.5)
        filters.append(
            f"[{music_idx}:a]volume={music_volume},"
            f"afade=t=in:st=0:d=1.5,afade=t=out:st={fade_out_start:.2f}:d=2.5[bg]"
        )
    if voice_idx is not None and music_idx is not None:
        filters.append(f"[{voice_idx}:a][bg]amix=inputs=2:duration=longest:normalize=0[mix]")
        audio_label = "[mix]"
    elif voice_idx is not None:
        audio_label = f"{voice_idx}:a"
    else:
        audio_label = "[bg]"

    if filters:
        cmd += ["-filter_complex", ";".join(filters)]
    cmd += [
        "-map", "0:v", "-map", audio_label,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-t", f"{duration:.3f}",
        "-movflags", "+faststart",
        str(tmp_out),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg audio mux failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
        )
    tmp_out.replace(video_path)
    return video_path
