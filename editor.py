"""
Video editing and assembly (moviepy v2 API).

Step 5: Concatenate scene clips with crossfade transitions.
Step 6: Burn in captions (scene text, timed per scene).
Step 7: Export final MP4, 1920x1080, H.264.

These functions accept any moviepy VideoClip, so they work identically
with real stock footage (from clip_finder) or placeholder clips (demo mode).
"""

from pathlib import Path

from moviepy import CompositeVideoClip, TextClip, VideoClip, vfx

VIDEO_SIZE = (1920, 1080)
FPS = 24
TRANSITION_DURATION = 0.4  # seconds of crossfade between scenes
CAPTION_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


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


def assemble_scenes(clips: list[VideoClip], transition: float = TRANSITION_DURATION) -> VideoClip:
    """
    Concatenate scene clips in order with a crossfade between each (Step 5).

    Each clip after the first starts `transition` seconds before the previous
    one ends and fades in over that overlap.
    """
    if not clips:
        raise ValueError("No clips to assemble")

    placed = [clips[0]]
    current_end = clips[0].duration

    for clip in clips[1:]:
        clip = clip.with_effects([vfx.CrossFadeIn(transition)])
        clip = clip.with_start(current_end - transition)
        placed.append(clip)
        current_end = clip.start + clip.duration

    return CompositeVideoClip(placed, size=clips[0].size)


def export_video(clip: VideoClip, output_path: str) -> str:
    """
    Export final video as MP4, H.264 (Step 7).
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    clip.write_videofile(
        output_path,
        fps=FPS,
        codec="libx264",
        audio=False,
        preset="faster",
        threads=4,
    )
    return output_path
