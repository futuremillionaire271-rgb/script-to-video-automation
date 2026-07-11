# Script → Video Studio

Turn a narration script (plus your own voiceover) into a fully edited,
documentary-style video — automatically. Matched stock B-roll, voice-locked
karaoke captions, cinematic grade, sound design, statement cards, and
typewriter keyword callouts, all generated from the text.

---

## What it does

Give it a **script** and (optionally) a **voiceover MP3**. It produces a
1920×1080 H.264 video where every line has matching footage, the captions
track your voice word-for-word, and the edit is built for retention.

### The pipeline

1. **Analyze** (`analyzer.py`) — turns each line into a *strong* stock-search
   query using a concept→visual dictionary (e.g. "caffeine" → *pouring hot
   coffee into cup*), suppresses abstract words, and carries the topic
   forward on abstract lines. Also picks one memorable **key term** per scene
   for the typewriter overlay.
2. **Voice alignment** (`align.py`) — transcribes the voiceover with Whisper
   and gets the true spoken time of every word, so captions and cuts lock to
   your actual voice (falls back to pause-detection if Whisper is offline).
3. **Scene split** (`scene_parser.py`) — groups sentences into ~6s scenes on
   real voice timings, never splitting a sentence.
4. **Find footage** (`clip_finder.py`) — searches Pexels + Pixabay, ranks by
   text relevance, and **vision-verifies** the pick with CLIP (`vision.py`)
   so the picture actually matches the line.
5. **Build each scene** (`scene_builder.py`) — download → trim → Ken Burns
   motion → karaoke caption → entity callout → typewriter keyword → edge
   transition → fast ffmpeg grade + vignette.
6. **Assemble** (`editor.py`) — ffmpeg concat (stream copy) + mux voiceover,
   ducked music, and the synthesized **sound-design** track (`sfx.py`).

### The edit (retention features)

- **Karaoke captions** — bold, ≤2 lines, phrase-chunked, each word
  highlighted in gold *as you say it* (Whisper-timed).
- **Typewriter keyword** — a memorable term (MELATONIN, MAGNESIUM…) types
  itself out in a monospace chip with a blinking cursor and **per-key
  mechanical keyboard clicks**, roughly once per scene.
- **Entity callouts** — place/person/number tags (ARIZONA, "TEN THIRTY").
- **Statement cards** — short power lines take the full screen in big type,
  with a riser→impact sound.
- **Motion + grade** — strong Ken Burns on every clip, teal-orange grade,
  vignette; crossfades within scenes, varied boundary transitions.
- **Sound design** — whooshes, impacts, risers synced to the cuts.

---

## Setup

```bash
pip install -r requirements.txt          # Python 3.10+
cp .env.example .env                      # add PEXELS_API_KEY + PIXABAY_API_KEY
```

First run downloads two models (Whisper `small.en`, CLIP `ViT-B-32`) from
HuggingFace; they are cached afterward. `ffmpeg` is provided via
`imageio-ffmpeg`, no system install needed.

---

## Usage

### Command line

```bash
# Full video from a script + your voiceover
python main.py my_scripts/sleep.txt --voiceover my_scripts/sleep.mp3 --scene-duration 6

# Quick style proof (first N scenes only)
python main.py my_scripts/sleep.txt --voiceover my_scripts/sleep.mp3 --limit 8

# See the auto-generated shot list without rendering (hand-edit, then --use-plan)
python main.py my_scripts/sleep.txt --voiceover my_scripts/sleep.mp3 --export-plan plan.json
python main.py my_scripts/sleep.txt --voiceover my_scripts/sleep.mp3 --use-plan plan.json
```

Output lands in `output/<script>_<timestamp>.mp4`. Runs are **checkpointed** —
re-run the same command to resume after an interruption.

### Web UI

```bash
python ui.py      # open http://localhost:8000
```

Paste a script, upload a voiceover, review the voice-aligned scene plan
(edit any scene's search phrase inline), hit **Render**, watch live progress,
and download finished videos.

### Options

| Flag | Default | Purpose |
|---|---|---|
| `--voiceover FILE` | — | Narration; captions + cuts lock to it via Whisper |
| `--scene-duration N` | 4.0 | Target scene length in seconds (6 recommended) |
| `--music FILE` | — | Background music, looped + ducked under narration |
| `--limit N` | — | Render only the first N scenes (style proof) |
| `--export-plan / --use-plan FILE` | — | Review / override per-scene search queries |
| `--audit` | off | Save each pick's thumbnail + match report, no render |
| `--no-typing` | off | Disable typewriter keyword + key-click sfx |
| `--no-cards` | off | Disable full-screen statement cards |
| `--no-sfx` | off | Disable sound design |
| `--no-callout` | off | Disable entity callout tags |
| `--no-vision` | off | Skip CLIP visual verification (faster) |
| `--no-motion` / `--no-grade` | off | Disable motion / grade+vignette |
| `--keep-temp` | off | Keep scene clips + checkpoint after export |
| `--workers N` | cores-1 (≤3) | Parallel scene render processes |

---

## Teaching it a new topic

`analyzer.py` maps ideas to shots via `CONCEPT_VISUALS` (regex → visual
phrase). To improve matches for a new subject, add rows there — most specific
patterns first — and add memorable terms to `KEY_TERMS` for the typewriter.

---

## Project structure

```
main.py           Orchestration, CLI, checkpointed render loop
analyzer.py       Script → strong search queries + key terms
align.py          Whisper word alignment (+ pause-based fallback)
scene_parser.py   Sentence-safe scene splitting (voice-aligned)
clip_finder.py    Pexels/Pixabay search, relevance + vision gating
vision.py         CLIP visual verification of candidates
scene_builder.py  Per-scene worker: build + treat + render
effects.py        Motion, transitions, boundary plan
captions.py       Karaoke phrase captions
callouts.py       Entity callout tags
kw_type.py        Typewriter keyword overlay
elements.py       Full-screen statement cards
sfx.py            Synthesized sound design (whoosh/impact/riser/key)
editor.py         ffmpeg concat, grade, audio mux
api_limits.py     Rate limiting, monthly budgets, search cache
pipeline_state.py Checkpoint + concurrency lock
ui.py             Premium web UI (FastAPI)
```

Everything runs locally. Stock from Pexels + Pixabay; alignment via Whisper;
verification via CLIP; sound design synthesized with numpy (no downloads, no
licensing).
