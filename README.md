# Script-to-Video Automation

Convert a text script into an edited video automatically with stock footage, captions, and transitions.

## Pipeline Overview

1. **Scene Splitting**: Break script into ~N-second scenes (default 4s, respecting sentence boundaries)
2. **Keyword Extraction**: Extract 2-3 visual keywords per scene
3. **Stock Footage Search**: Find matching clips via Pexels/Pixabay APIs (alternating per scene, rate-limited, results cached 24h)
4. **Download & Trim**: Download clips, trim to scene duration, delete raw files immediately
5. **Assemble**: ffmpeg concat demuxer for the final stitch (fast at any scale); short fade or hard-cut transitions baked per scene
6. **Captions**: Burn scene text as on-screen captions
7. **Export**: Output final MP4 (1920x1080, H.264)

Built to scale to 30+ minute videos (450+ scenes): API throttling, search caching,
per-scene disk cleanup, and checkpoint-based resume after interruption.

## Project Structure

```
.
├── main.py                    # Entry point / orchestration
├── scene_parser.py            # Steps 1-2: Scene splitting & keyword extraction
├── clip_finder.py             # Steps 3-4: Stock footage search & download
├── editor.py                  # Steps 5-7: Assembly, captions, export
├── api_limits.py              # Rate limiting, monthly budgets, search cache
├── pipeline_state.py          # Checkpointing for resumable runs
├── test_scenes.py             # Test harness for scene breakdown
├── test_data/
│   └── sample_script.txt      # Sample 78-second script for testing
├── cache/                     # Search cache + API usage counters (created on first run)
├── temp/                      # Scene clips + progress.json during a run
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

### Run Full Pipeline

```bash
python main.py your_script.txt
```

Options:

| Flag | Default | Purpose |
|---|---|---|
| `--demo` | off | Placeholder visuals, no API keys/network needed |
| `--scene-duration N` | 4.0 | Target scene length in seconds (6-8 recommended for 30+ min videos) |
| `--transition fade\|cut` | fade | 0.2s fade to/from black at scene edges, or hard cuts |
| `--progress-every N` | 10 | Progress line every N completed scenes |
| `--keep-temp` | off | Keep scene clips + checkpoint after export |

Outputs final video to `output/your_script_<timestamp>.mp4`.

**Interrupted?** Just re-run the same command — progress is checkpointed to
`temp/progress.json` after every scene, and completed scenes are skipped.
(The checkpoint resets automatically if the script text or settings change.)

## Implementation Status

- ✅ **Step 1**: Scene splitting (respects sentence boundaries, configurable target length)
- ✅ **Step 2**: Keyword extraction (nouns + adjectives, rule-based)
- ✅ **Step 3**: Stock footage search (Pexels + Pixabay alternating, throttled, cached, broadening queries)
- ✅ **Step 4**: Download & trim (streamed download, trim/loop to scene duration, fit to 1080p, raw files deleted per scene)
- ✅ **Step 5**: Assembly via ffmpeg concat demuxer (stream copy, no re-encode)
- ✅ **Step 6**: Captions burned in per scene
- ✅ **Step 7**: MP4 export (1920x1080, H.264)
- ✅ **Scale**: Rate limiting (Pexels 200/hr + 20k/month, Pixabay 100/60s), 24h search cache, resumable checkpoints, progress reporting

## Design Notes

- **Modular**: Each step is independent, so you can test/swap/debug pieces separately
- **Sentence Boundaries**: Scenes never split mid-sentence; long sentences may extend beyond the target
- **Speaking Pace**: Assumes 2.5 words/second for scene duration estimation
- **Keywords**: Extracted via NLTK POS tagging; can be improved with semantic analysis later
- **Temp Files**: Raw downloads are deleted per scene; trimmed scene clips are cleaned up
  after a successful export unless `--keep-temp` is passed
- **Transitions**: The concat demuxer can't overlap clips, so true crossfades are replaced
  by short fades baked into each scene's edges (`--transition cut` for hard cuts)
- **Clip Variety**: Clips aren't reused across scenes while unused candidates remain;
  on very long runs with narrow topics, the best already-used clip is reused (logged)
  rather than failing

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
