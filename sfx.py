"""
Synthesized sound design — audio punctuation for retention editing.

"Sound moves attention faster than picture": a whoosh makes a cut feel
physical, a riser signals something is coming, an impact lands an emphasis
moment. All effects are synthesized with numpy (filtered noise + sine
envelopes), so there are no downloads and no licensing questions.

Events are placed at scene boundaries by type:
  punch/zoompunch boundary -> impact
  white flash boundary     -> whoosh
  dip-to-black boundary    -> soft whoosh
  statement card           -> riser (leading in) + impact (on land)
"""

import wave
from pathlib import Path

import numpy as np

SR = 44100
# Mix levels relative to the voiceover. Peaks sit just under speech peaks
# (~-7 dBFS): clearly FELT at boundaries without masking a single word.
GAIN = {"whoosh": 0.30, "whoosh_soft": 0.18, "impact": 0.45, "riser": 0.25,
        "card": 0.45, "key": 0.32}
MIN_GAP = 1.2  # never stack effects closer than this (seconds)


def _lowpass(x: np.ndarray, k: int) -> np.ndarray:
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="same")


def _whoosh(dur=0.45) -> np.ndarray:
    n = int(SR * dur)
    t = np.linspace(0, 1, n)
    noise = np.random.RandomState(7).randn(n)
    # Sweep: heavy lowpass opening up then closing (approximates a doppler swish)
    a = _lowpass(noise, 261) * (1 - t)
    b = _lowpass(noise, 31) * t
    x = (a + b) * np.sin(np.pi * t) ** 1.5
    return x / (np.abs(x).max() + 1e-9)


def _impact(dur=0.7) -> np.ndarray:
    n = int(SR * dur)
    t = np.linspace(0, dur, n)
    thump = np.sin(2 * np.pi * (55 - 20 * t) * t) * np.exp(-t * 7)
    click = np.random.RandomState(3).randn(n) * np.exp(-t * 60) * 0.4
    x = thump + _lowpass(click, 9)
    return x / (np.abs(x).max() + 1e-9)


def _riser(dur=0.9) -> np.ndarray:
    n = int(SR * dur)
    t = np.linspace(0, 1, n)
    noise = np.random.RandomState(11).randn(n)
    x = _lowpass(noise, 41) * t ** 2.2          # swell
    x += 0.35 * np.sin(2 * np.pi * (180 + 240 * t**2) * t * dur) * t ** 2  # rising tone
    x *= np.minimum(1.0, (1 - t) * 20 + 0.0)    # hard stop at the cut
    return x / (np.abs(x).max() + 1e-9)


def _key_click(dur=0.055) -> np.ndarray:
    """A single mechanical keyboard keypress: a sharp click + tiny body thock."""
    n = int(SR * dur)
    t = np.linspace(0, dur, n)
    rng = np.random.RandomState(19)
    click = rng.randn(n) * np.exp(-t * 380)                 # crisp attack
    thock = np.sin(2 * np.pi * 130 * t) * np.exp(-t * 90) * 0.5  # low body
    x = _lowpass(click, 3) + thock
    return x / (np.abs(x).max() + 1e-9)


def _card_hit(riser_dur=0.9, impact_dur=0.7) -> np.ndarray:
    """Riser building into an impact — the statement-card landing."""
    r = _riser(riser_dur) * (GAIN["riser"] / GAIN["card"])
    i = _impact(impact_dur) * (GAIN["impact"] / GAIN["card"])
    x = np.concatenate([r, i])
    return x / (np.abs(x).max() + 1e-9)


_BANK = {"whoosh": _whoosh, "whoosh_soft": _whoosh, "impact": _impact,
         "riser": _riser, "card": _card_hit, "key": _key_click}


def build_sfx_track(events: list[tuple[str, float]], duration: float,
                    out_path: Path) -> Path | None:
    """
    Place effects on a timeline and write one mono WAV covering the video.
    events: [(type, time_seconds)] — 'riser' times mean the moment the riser
    LANDS (it is placed so it ends at that time). Returns None if no events.
    """
    if not events:
        return None
    total = int(SR * (duration + 1.0))
    track = np.zeros(total, dtype=np.float32)

    last_t = -1e9
    for kind, t in sorted(events, key=lambda e: e[1]):
        if kind not in _BANK:
            continue
        if kind != "key" and t - last_t < MIN_GAP:
            continue
        x = _BANK[kind]() * GAIN[kind]
        if kind == "riser":
            start_t = t - len(x) / SR
        elif kind == "card":
            start_t = t - 0.9  # riser portion leads in; impact lands at t
        else:
            start_t = t - 0.06
        i0 = max(0, int(start_t * SR))
        i1 = min(total, i0 + len(x))
        if i1 > i0:
            track[i0:i1] += x[: i1 - i0]
            if kind != "key":
                last_t = t

    track = np.clip(track, -0.9, 0.9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (track * 32767).astype(np.int16)
    with wave.open(str(out_path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return out_path


def plan_sfx_events(scenes, boundaries, card_flags) -> list[tuple[str, float]]:
    """Derive the audio-punctuation plan from the visual plan."""
    events: list[tuple[str, float]] = []
    n = len(scenes)
    for i in range(n - 1):
        t = scenes[i + 1].start_time
        style = boundaries[i] if boundaries and i < len(boundaries) else "cut"
        if card_flags[i + 1]:
            events.append(("card", t))
        elif style in ("punch", "zoompunch"):
            events.append(("impact", t))
        elif style == "white":
            events.append(("whoosh", t))
        elif style == "black":
            events.append(("whoosh_soft", t))
    return events
