"""
Main entry point for script-to-video automation.

Usage:
    python main.py <script_file> [--keep-temp]

Pipeline:
    1. Scene splitting: Break script into ~4-second scenes.
    2. Keyword extraction: Extract visual keywords per scene.
    3. Stock footage search: Find clips for each scene (Pexels/Pixabay).
    4. Download & trim: Download and trim clips to scene duration.
    5. Assemble: Concatenate with transitions.
    6. Captions: Burn in scene text as captions.
    7. Export: Output final MP4.
"""

import argparse
import sys
from pathlib import Path
from scene_parser import load_script, split_into_scenes, print_scenes


def main():
    parser = argparse.ArgumentParser(
        description='Convert a text script into an edited video automatically.'
    )
    parser.add_argument('script_file', help='Path to input script (text file)')
    parser.add_argument(
        '--keep-temp',
        action='store_true',
        help='Keep intermediate/temp files after export (default: clean up)'
    )

    args = parser.parse_args()

    # Load and parse script
    script_path = Path(args.script_file)
    if not script_path.exists():
        print(f"Error: Script file not found: {script_path}")
        sys.exit(1)

    print(f"Loading script: {script_path}")
    script = load_script(str(script_path))

    print("Splitting into scenes and extracting keywords...")
    scenes = split_into_scenes(script)

    print_scenes(scenes)

    print(f"Keep temp files: {args.keep_temp}")
    print("\nSteps 1-2 (scene splitting & keyword extraction) complete.")
    print("Next: Implement Steps 3-7 (clip search, download, assembly, captions, export)")


if __name__ == '__main__':
    main()
