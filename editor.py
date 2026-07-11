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

# Teal-orange cinematic grade + vignette, done as ffmpeg C filters instead
# of per-frame numpy — roughly 350ms/frame -> ~free. Approximates effects.apply_grade.
_GRADE_VF = ("colorbalance=rs=-0.05:gs=0.02:bs=0.06:rh=0.07:gh=0.03:bh=-0.05,"
             "eq=contrast=1.12:saturation=1.15,vignette=angle=PI/5")


def render_scene_file(clip: VideoClip, path: Path, grade: bool = False) -> Path:
    """
    Encode one finished scene to disk with the uniform settings that make
    lossless final concat possible. When `grade` is set, the cinematic
    grade + vignette are applied here as fast ffmpeg filters (a second,
    quick encode) instead of slow per-frame numpy in the moviepy pipeline.
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
    if grade:
        graded = path.with_name(path.stem + ".graded.mp4")
        cmd = [get_ffmpeg_exe(), "-y", "-i", str(path), "-vf", _GRADE_VF,
               "-c:v", "libx264", "-preset", "veryfast", *_SCENE_ENCODE_ARGS,
               "-an", "-loglevel", "error", str(graded)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            graded.replace(path)
        # if grading fails, keep the ungraded scene rather than crash the run
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
              music: Path = None, voiceover: Path = None, sfx: Path = None,
              music_volume: float = MUSIC_VOLUME) -> Path:
    """
    Add audio to the assembled video without re-encoding the video stream.

    - voiceover plays at full volume (the star of the mix)
    - music is looped to cover the video, ducked to `music_volume`, with a
      fade-in and a fade-out at the end
    - sfx is the synthesized sound-design track (whooshes/impacts/risers),
      pre-levelled, mixed as-is
    """
    if music is None and voiceover is None and sfx is None:
        return video_path

    tmp_out = video_path.with_name(video_path.stem + ".audio.mp4")
    cmd = [get_ffmpeg_exe(), "-y", "-i", str(video_path)]

    music_idx = voice_idx = sfx_idx = None
    next_idx = 1
    if music is not None:
        cmd += ["-stream_loop", "-1", "-i", str(music)]
        music_idx, next_idx = next_idx, next_idx + 1
    if voiceover is not None:
        cmd += ["-i", str(voiceover)]
        voice_idx, next_idx = next_idx, next_idx + 1
    if sfx is not None:
        cmd += ["-i", str(sfx)]
        sfx_idx = next_idx

    filters = []
    mix_inputs = []
    if music_idx is not None:
        fade_out_start = max(0.0, duration - 2.5)
        filters.append(
            f"[{music_idx}:a]volume={music_volume},"
            f"afade=t=in:st=0:d=1.5,afade=t=out:st={fade_out_start:.2f}:d=2.5[bg]"
        )
        mix_inputs.append("[bg]")
    if voice_idx is not None:
        mix_inputs.insert(0, f"[{voice_idx}:a]")
    if sfx_idx is not None:
        mix_inputs.append(f"[{sfx_idx}:a]")

    if len(mix_inputs) > 1:
        filters.append("".join(mix_inputs)
                       + f"amix=inputs={len(mix_inputs)}:duration=longest:normalize=0[mix]")
        audio_label = "[mix]"
    else:
        audio_label = mix_inputs[0].strip("[]") if ":" in mix_inputs[0] else mix_inputs[0]

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
