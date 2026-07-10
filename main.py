"""
Main entry point for script-to-video automation.

Usage:
    python main.py <script_file> [--demo] [--keep-temp]
                   [--scene-duration N] [--transition fade|cut]
                   [--progress-every N]

Pipeline:
    1. Scene splitting: Break script into ~N-second scenes (default 4).
    2. Keyword extraction: Extract visual keywords per scene.
    3. Stock footage search: Pexels/Pixabay, alternating per scene,
       throttled to provider rate limits, results cached 24h.
    4. Download & trim: Per scene — download, trim, render, then delete the
       raw download immediately.
    5. Assemble: ffmpeg concat demuxer (stream copy — fast at any scale).
    6. Captions: Burned into each scene file, timed per scene.
    7. Export: Final MP4 (1920x1080, H.264).

Progress is checkpointed to temp/progress.json after every scene, so an
interrupted run resumes from the last completed scene.

--demo skips Steps 3-4 (no API keys needed) and uses generated placeholder
visuals per scene.
"""

import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from scene_parser import load_script, split_into_scenes, print_scenes
from pipeline_state import Checkpoint, fingerprint

TEMP_DIR = Path("temp")


def build_output_path(script_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("output") / f"{script_path.stem}_{timestamp}.mp4"


def _format_eta(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def main():
    parser = argparse.ArgumentParser(
        description='Convert a text script into an edited video automatically.'
    )
    parser.add_argument('script_file', help='Path to input script (text file)')
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Use generated placeholder visuals instead of stock footage '
             '(no API keys required)'
    )
    parser.add_argument(
        '--keep-temp',
        action='store_true',
        help='Keep intermediate/temp files after export (default: clean up)'
    )
    parser.add_argument(
        '--scene-duration',
        type=float,
        default=4.0,
        metavar='SECONDS',
        help='Target scene length in seconds (default: 4). For 30+ minute '
             'videos, 6-8s reduces API load and footage repetition.'
    )
    parser.add_argument(
        '--transition',
        choices=('fade', 'cut'),
        default='fade',
        help='Scene transition: short fade to/from black at scene edges, '
             'or clean hard cuts (default: fade)'
    )
    parser.add_argument(
        '--progress-every',
        type=int,
        default=10,
        metavar='N',
        help='Print a progress line every N completed scenes (default: 10)'
    )

    args = parser.parse_args()

    script_path = Path(args.script_file)
    if not script_path.exists():
        print(f"Error: Script file not found: {script_path}")
        sys.exit(1)

    # Steps 1-2: scenes + keywords
    print(f"Loading script: {script_path}")
    script = load_script(str(script_path))

    print("Splitting into scenes and extracting keywords...")
    scenes = split_into_scenes(script, target_duration=args.scene_duration)
    print_scenes(scenes, limit=10 if len(scenes) > 40 else None)

    # Resumable checkpoint, keyed to this script + settings
    run_id = fingerprint(script, len(scenes), args.scene_duration)
    checkpoint = Checkpoint(TEMP_DIR / "progress.json", run_id)
    already_done = sum(
        1 for i in range(len(scenes)) if checkpoint.done_path(i) is not None
    )
    if already_done:
        print(f"Resuming: {already_done}/{len(scenes)} scenes already complete.\n")

    from editor import add_caption, apply_transition, render_scene_file, concat_scene_files

    if args.demo:
        print("Demo mode: generating placeholder visuals (Steps 3-4 skipped)...")
    else:
        from clip_finder import ClipSearchError, check_api_access, fetch_scene_clip
        print("Preflight: checking API keys and connectivity...")
        try:
            check_api_access()
        except ClipSearchError as exc:
            print(f"\nFATAL: {exc}")
            sys.exit(1)
        print("Searching and downloading stock footage (Pexels + Pixabay, alternating)...")

    used_ids = set(checkpoint.used_clips)
    scenes_dir = TEMP_DIR / "scenes"
    raw_dir = TEMP_DIR / "raw"
    scene_files: list[Path] = []

    run_start = time.monotonic()
    rendered_this_run = 0

    for i, scene in enumerate(scenes):
        existing = checkpoint.done_path(i)
        if existing is not None:
            scene_files.append(existing)
            continue

        scene_file = scenes_dir / f"scene_{i + 1:04d}.mp4"
        raw_path = None

        if args.demo:
            from clip_finder import make_placeholder_clip
            clip = make_placeholder_clip(scene, i)
        else:
            clip, raw_path = fetch_scene_clip(scene, i, raw_dir, used_ids)

        clip = add_caption(clip, scene.text)
        clip = apply_transition(clip, args.transition)
        render_scene_file(clip, scene_file)
        clip.close()

        # Per-scene cleanup: raw download gone as soon as the trimmed,
        # captioned scene file is safely on disk
        if raw_path is not None:
            raw_path.unlink(missing_ok=True)

        checkpoint.mark_done(i, scene_file, used_ids)
        scene_files.append(scene_file)
        rendered_this_run += 1

        if rendered_this_run % args.progress_every == 0 or i == len(scenes) - 1:
            elapsed = time.monotonic() - run_start
            remaining = len(scenes) - (i + 1)
            per_scene = elapsed / rendered_this_run
            print(f"Scene {i + 1}/{len(scenes)} complete "
                  f"| {per_scene:.1f}s/scene | ETA {_format_eta(remaining * per_scene)}")

    # Steps 5+7: fast final stitch (no re-encode)
    output_path = build_output_path(script_path)
    print(f"\nAssembling {len(scene_files)} scenes with ffmpeg concat -> {output_path}")
    concat_scene_files(scene_files, output_path)

    # Clean up scene clips + checkpoint after a successful export
    if not args.keep_temp and TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
        print("Cleaned up temp files (use --keep-temp to keep them).")
    elif args.keep_temp and TEMP_DIR.exists():
        print(f"Temp files kept in {TEMP_DIR}/")

    print(f"\nDone: {output_path}")


if __name__ == '__main__':
    main()
