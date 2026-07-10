"""
Worker-side scene construction: download shots, apply the documentary
treatment, and render one scene file.

Runs inside worker processes (see --workers). Everything a job needs is a
plain dict — clip *selection* happens in the main process (single-threaded
rate limiting + no-repeat tracking), workers only download and render.
"""

from pathlib import Path

from moviepy import VideoFileClip, concatenate_videoclips, vfx

from clip_finder import ClipCandidate, download_clip, _fit_to_frame, make_placeholder_clip
from editor import add_caption, render_scene_file
from effects import add_vignette, apply_edges, apply_grade, apply_motion, punch_in
from scene_parser import Scene


def render_scene_job(job: dict) -> int:
    """
    Build and render one scene from a job dict:
      index, n_scenes, text, keywords, start, end,
      shots: [{candidate: {...}, duration: float}] or None for demo mode,
      in_style, out_style, motion (style or None), grade (bool),
      raw_dir, scene_file
    Returns the scene index on success.
    """
    scene = Scene(job["text"], job["start"], job["end"], job["keywords"])
    raw_paths: list[Path] = []
    sources = []

    if job["shots"] is None:
        clip = make_placeholder_clip(scene, job["index"])
    else:
        for n, shot in enumerate(job["shots"]):
            candidate = ClipCandidate(**shot["candidate"])
            raw = Path(job["raw_dir"]) / (
                f"raw_{job['index'] + 1:04d}_{n}_{candidate.source}_{candidate.video_id}.mp4"
            )
            download_clip(candidate, raw)
            raw_paths.append(raw)

            c = VideoFileClip(str(raw)).without_audio()
            d = shot["duration"]
            c = c.subclipped(0, d) if c.duration >= d else c.with_effects([vfx.Loop(duration=d)])
            sources.append(_fit_to_frame(c))
        clip = sources[0] if len(sources) == 1 else concatenate_videoclips(sources)

    # Documentary treatment: motion -> grade -> caption -> edge transitions
    in_style = job["in_style"]
    if in_style == "punch":
        clip = punch_in(clip)
        in_style = "cut"
    elif job["motion"]:
        clip = apply_motion(clip, job["motion"])

    if job["grade"]:
        clip = add_vignette(apply_grade(clip))

    clip = add_caption(clip, scene.text, scene.keywords)
    clip = apply_edges(clip, in_style, job["out_style"])

    render_scene_file(clip, Path(job["scene_file"]))
    clip.close()
    for c in sources:
        c.close()

    # Per-scene cleanup: raw downloads deleted as soon as the rendered
    # scene file is safely on disk
    for raw in raw_paths:
        raw.unlink(missing_ok=True)

    return job["index"]
