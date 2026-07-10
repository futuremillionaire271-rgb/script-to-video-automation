"""
Video editing and assembly (moviepy v2 API + ffmpeg concat demuxer).

Step 5: Assemble scene clips. At scale (hundreds of scenes) the final stitch
        uses ffmpeg's concat demuxer with stream copy — near-instant and
        flat-memory — instead of moviepy. Every scene file is rendered with
        identical codec settings so lossless concat is valid.
Step 6: Burn in captions (scene text, timed per scene).
Step 7: Export final MP4, 1920x1080, H.264.

Transitions: the concat demuxer can only butt clips together, so true
crossfades are replaced by a short fade to/from black baked into each scene
file's edges (--transition fade), or clean hard cuts (--transition cut).
"""

import subprocess
from pathlib import Path

from imageio_ffmpeg import get_ffmpeg_exe
from moviepy import CompositeVideoClip, TextClip, VideoClip, vfx

VIDEO_SIZE = (1920, 1080)
FPS = 24
FADE_DURATION = 0.2  # seconds faded to/from black at each scene edge
CAPTION_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Uniform encoder settings for every scene file — REQUIRED for the concat
# demuxer's stream copy to produce a valid output
_SCENE_ENCODE_ARGS = ["-crf", "23", "-pix_fmt", "yuv420p"]


def add_caption(clip: VideoClip, text: str) -> VideoClip:
    """
    Burn caption text onto the bottom of a clip (Step 6).

    Caption spans the clip's full duration — timing per scene comes free
    because each scene clip is exactly the scene's duration.
    """
    width, height = clip.size

    caption = TextClip(
        font=CAPTION_FONT,
        text=text,
        font_size=int(height * 0.042),
        color="white",
        stroke_color="black",
        stroke_width=2,
        method="caption",
        size=(int(width * 0.85), None),
        text_align="center",
    )
    caption = caption.with_duration(clip.duration).with_position(
        ("center", height - caption.h - int(height * 0.07))
    )

    return CompositeVideoClip([clip, caption], size=clip.size)


def apply_transition(clip: VideoClip, style: str = "fade") -> VideoClip:
    """Bake the scene-edge transition into the clip ('fade' or 'cut')."""
    if style == "fade":
        return clip.with_effects([vfx.FadeIn(FADE_DURATION), vfx.FadeOut(FADE_DURATION)])
    return clip


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
    Final assembly (Step 5 + 7) via ffmpeg's concat demuxer with stream
    copy: no re-encode, near-instant, flat memory — unlike moviepy, which
    gets slow and memory-heavy with hundreds of clips.
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
