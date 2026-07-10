"""
Test script to verify scene splitting and keyword extraction.

Run with: python test_scenes.py
"""

from pathlib import Path
from scene_parser import load_script, split_into_scenes, print_scenes


def test_sample_script():
    """Test scene splitting on sample script."""
    script_path = Path('test_data/sample_script.txt')

    if not script_path.exists():
        print(f"Error: Test script not found at {script_path}")
        return

    print(f"Testing with: {script_path}\n")
    script = load_script(str(script_path))

    print(f"Script length: {len(script)} characters, {len(script.split())} words")
    print(f"Estimated duration: {len(script.split()) / 2.5:.1f}s\n")

    scenes = split_into_scenes(script)
    print_scenes(scenes)

    # Verify keyword extraction
    print("Keyword extraction verification:")
    print("-" * 80)
    for i, scene in enumerate(scenes, 1):
        print(f"\nScene {i} keywords: {scene.keywords}")
        print(f"  Scene text (first 60 chars): {scene.text[:60]}...")


if __name__ == '__main__':
    test_sample_script()
