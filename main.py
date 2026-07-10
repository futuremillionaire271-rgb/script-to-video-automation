"""
Main entry point for script-to-video automation.

Usage:
    python main.py <script_file> [--demo] [--keep-temp]

Pipeline:
    1. Scene splitting: Break script into ~4-second scenes.
    2. Keyword extraction: Extract visual keywords per scene.
    3. Stock footage search: Find clips for each scene (Pexels/Pixabay).
    4. Download & trim: Download and trim clips to scene duration.
    5. Assemble: Concatenate with crossfade transitions.
    6. Captions: Burn in scene text as captions.
    7. Export: Output final MP4 (1920x1080, H.264).

--demo skips Steps 3-4 (no API keys needed) and uses generated placeholder
visuals per scene, so you can preview scene timing, captions, and transitions.
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

from scene_parser import load_script, split_into_scenes, print_scenes


def build_output_path(script_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("output") / f"{script_path.stem}_{timestamp}.mp4"


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

    args = parser.parse_args()

    script_path = Path(args.script_file)
    if not script_path.exists():
        print(f"Error: Script file not found: {script_path}")
        sys.exit(1)

    # Steps 1-2: scenes + keywords
    print(f"Loading script: {script_path}")
    script = load_script(str(script_path))

    print("Splitting into scenes and extracting keywords...")
    scenes = split_into_scenes(script)
    print_scenes(scenes)

    # Steps 3-4: source a clip per scene
    if args.demo:
        from clip_finder import make_placeholder_clip
        print("Demo mode: generating placeholder visuals (Steps 3-4 skipped)...")
        scene_clips = [
            make_placeholder_clip(scene, i) for i, scene in enumerate(scenes)
        ]
    else:
        from clip_finder import ClipSearchError, check_api_access, prepare_scene_clip
        print("Preflight: checking API keys and connectivity...")
        try:
            check_api_access()
        except ClipSearchError as exc:
            print(f"\nFATAL: {exc}")
            sys.exit(1)
        print("Searching and downloading stock footage (Pexels, Pixabay fallback)...")
        scene_clips = [
            prepare_scene_clip(scene, i) for i, scene in enumerate(scenes)
        ]

    # Step 6: burn captions onto each scene clip (per-scene timing is implicit)
    from editor import add_caption, assemble_scenes, export_video
    print("Burning captions...")
    scene_clips = [
        add_caption(clip, scene.text) for clip, scene in zip(scene_clips, scenes)
    ]

    # Step 5: assemble with crossfades
    print("Assembling scenes with crossfade transitions...")
    final = assemble_scenes(scene_clips)

    # Step 7: export
    output_path = build_output_path(script_path)
    print(f"Exporting to {output_path} ...")
    export_video(final, str(output_path))

    # Clean up downloaded/trimmed clips unless asked to keep them
    temp_dir = Path("temp")
    if not args.keep_temp and temp_dir.exists():
        shutil.rmtree(temp_dir)
        print("Cleaned up temp files (use --keep-temp to keep them).")
    elif args.keep_temp and temp_dir.exists():
        print(f"Temp files kept in {temp_dir}/")

    print(f"\nDone: {output_path}")


if __name__ == '__main__':
    main()
