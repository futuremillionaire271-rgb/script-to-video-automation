"""
Voiceover alignment — model-free.

Captions drift when scene timing is *estimated* from word counts: real
narration has pauses between sentences that estimates can't see, and the
error compounds over minutes. This module analyzes the actual voiceover
waveform, finds the silences (pauses), and snaps sentence/scene boundaries
to them, so cuts and captions track the recorded voice.

How:
1. Decode the audio to mono PCM (ffmpeg) and compute frame RMS energy.
2. Threshold -> speech vs silence segments.
3. Distribute the script's sentences across SPEECH TIME (not wall time),
   proportionally to word count — so pauses no longer push captions late.
4. Snap each sentence boundary to the nearest detected pause near its
   expected position (sentence ends almost always coincide with a pause).

When word-level ASR (Whisper) is available, it can replace step 3-4 with
true per-word timestamps; the interface stays the same.
"""

import subprocess

import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe

SR = 16000
FRAME = int(SR * 0.02)        # 20 ms frames
SILENCE_DB = -38.0            # below this (rel. dBFS) counts as silence
MIN_SILENCE = 0.28            # pause length that separates speech segments
SNAP_WINDOW = 1.1             # how far a boundary may move to reach a pause


def _load_mono(audio_path: str) -> np.ndarray:
    """Decode audio to 16 kHz mono float32 via ffmpeg."""
    cmd = [get_ffmpeg_exe(), "-i", audio_path, "-f", "s16le", "-acodec",
           "pcm_s16le", "-ac", "1", "-ar", str(SR), "-loglevel", "error", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def detect_silences(audio_path: str) -> tuple[list[tuple[float, float]], float]:
    """Return ([(silence_start, silence_end), ...], total_duration)."""
    samples = _load_mono(audio_path)
    n_frames = len(samples) // FRAME
    frames = samples[: n_frames * FRAME].reshape(n_frames, FRAME)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    # Adaptive floor: quiet recordings shouldn't read as all-silence
    threshold = max(SILENCE_DB, np.percentile(db, 92) - 26.0)
    quiet = db < threshold

    silences = []
    start = None
    for i, q in enumerate(quiet):
        t = i * FRAME / SR
        if q and start is None:
            start = t
        elif not q and start is not None:
            if t - start >= MIN_SILENCE:
                silences.append((start, t))
            start = None
    duration = len(samples) / SR
    if start is not None and duration - start >= MIN_SILENCE:
        silences.append((start, duration))
    return silences, duration


def align_sentences(sentence_word_counts: list[int], audio_path: str) -> list[tuple[float, float]]:
    """
    Assign each sentence a (start, end) in the voiceover, using pause
    structure. Sentences tile the full audio: sentence k ends where k+1
    starts, first starts at 0, last ends at audio end.
    """
    silences, duration = detect_silences(audio_path)

    # Build the speech-time map: cumulative speech seconds at wall time t.
    # Boundaries placed in speech-time then mapped back ignore pause length.
    events = []  # (wall_time, is_silence_start)
    for s, e in silences:
        events.append((s, e))
    total_silence = sum(e - s for s, e in silences)
    total_speech = max(duration - total_silence, 1e-6)

    def speech_to_wall(target_speech: float) -> float:
        spoken = 0.0
        cursor = 0.0
        for s, e in silences:
            chunk = s - cursor  # speech before this silence
            if spoken + chunk >= target_speech:
                return cursor + (target_speech - spoken)
            spoken += chunk
            cursor = e
        return min(cursor + (target_speech - spoken), duration)

    # Pause midpoints are the natural snap targets for sentence boundaries
    pause_mids = [(s + e) / 2 for s, e in silences]

    total_words = max(sum(sentence_word_counts), 1)
    bounds = [0.0]
    cum_words = 0
    for wc in sentence_word_counts[:-1]:
        cum_words += wc
        expected = speech_to_wall(cum_words / total_words * total_speech)
        # Snap to the nearest pause midpoint if one is close enough
        snapped = expected
        best = SNAP_WINDOW
        for mid in pause_mids:
            d = abs(mid - expected)
            if d < best and mid > bounds[-1] + 0.4:
                best = d
                snapped = mid
        bounds.append(max(snapped, bounds[-1] + 0.3))
    bounds.append(duration)

    return [(bounds[i], bounds[i + 1]) for i in range(len(sentence_word_counts))]
