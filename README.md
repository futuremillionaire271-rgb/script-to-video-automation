# Script-to-Video Automation

Convert a text script into an edited video automatically with stock footage, captions, and transitions.

## Pipeline Overview

1. **Scene Splitting**: Break script into ~4-second scenes (respecting sentence boundaries)
2. **Keyword Extraction**: Extract 2-3 visual keywords per scene
3. **Stock Footage Search**: Find matching clips via Pexels/Pixabay APIs
4. **Download & Trim**: Download clips and trim to scene duration
5. **Assemble**: Concatenate clips with 0.3-0.5s crossfade transitions
6. **Captions**: Burn scene text as on-screen captions
7. **Export**: Output final MP4 (1920x1080, H.264)

## Project Structure

```
.
├── main.py                    # Entry point
├── scene_parser.py            # Steps 1-2: Scene splitting & keyword extraction
├── clip_finder.py             # Steps 3-4: Stock footage search & download
├── editor.py                  # Steps 5-7: Assembly, captions, export
├── test_scenes.py             # Test harness for scene breakdown
├── test_data/
│   └── sample_script.txt      # Sample 78-second script for testing
├── output/                    # Final video exports (created on first run)
└── requirements.txt           # Python dependencies
```

## Quick Start

### Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure API keys (get from [Pexels](https://www.pexels.com/api/) and [Pixabay](https://pixabay.com/api/docs/)):
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

### Test Scene Splitting

Verify the scene breakdown on the sample script:
```bash
python test_scenes.py
```

This outputs:
- 15 scenes from ~78-second sample script
- Each scene's text, duration, and visual keywords
- Total estimated video duration

### Run Full Pipeline (WIP)

Once all steps are implemented:
```bash
python main.py test_data/sample_script.txt
# or
python main.py your_script.txt --keep-temp
```

Outputs final video to `output/your_script_<timestamp>.mp4`

## Implementation Status

- ✅ **Step 1**: Scene splitting (respects sentence boundaries, targets ~4s)
- ✅ **Step 2**: Keyword extraction (nouns + adjectives, rule-based)
- ⏳ **Steps 3-7**: To be implemented (clip search, download, assembly, captions, export)

## Design Notes

- **Modular**: Each step is independent, so you can test/swap/debug pieces separately
- **Sentence Boundaries**: Scenes never split mid-sentence; long sentences may extend beyond 4s
- **Speaking Pace**: Assumes 2.5 words/second for scene duration estimation
- **Keywords**: Extracted via NLTK POS tagging; can be improved with semantic analysis later
- **Temp Files**: Intermediate clips kept by default (enable `--keep-temp` to auto-clean after export)

## Dependencies

- `moviepy`: Video editing (FFmpeg-backed)
- `requests`: HTTP for API calls
- `python-dotenv`: Environment variable management
- `nltk`: NLP for keyword extraction
- `Pillow`: Image processing

## Future Improvements

- More sophisticated keyword extraction (semantic analysis, named entities)
- Better clip matching scoring (prefer landscape, HD, reasonable duration)
- Speech synthesis for narration overlay
- Dynamic caption sizing/positioning
- Title cards between scenes
- Multiple stock footage sources (YouTube, Unsplash, etc.)
