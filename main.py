"""
Main entry point for script-to-video automation.

Usage:
    python main.py <script_file> [options]

Pipeline:
    1. Scene splitting: ~N-second scenes (default 4), sentence-safe. Pace is
       auto-calibrated from --voiceover audio when provided.
    2. Keyword extraction: visual keywords per scene (hand-editable via
       --export-plan / --use-plan).
    3. Stock footage search: Pexels/Pixabay alternating, throttled, cached.
    4. Download & trim: per shot — scenes longer than --max-shot are built
       from multiple different clips of the same subject, so visuals keep
       moving without splitting the caption. Raw files deleted immediately.
    5. Assemble: documentary treatment baked per scene — Ken Burns motion,
       varied boundary transitions (cuts, dips, white flash, punch-ins),
       light grade + vignette — then ffmpeg concat (stream copy).
    6. Captions: burned in per scene, key words highlighted in gold.
    7. Export: 1920x1080 H.264; background music ducked under an optional
       voiceover track, muxed without re-encoding video.

Progress is checkpointed after every scene; interrupted runs resume.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from scene_parser import load_script, split_into_scenes, print_scenes
from pipeline_state import Checkpoint, PipelineLock, fingerprint

TEMP_DIR = Path("temp")
MAX_SHOTS_PER_SCENE = 3


def build_output_path(script_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("output") / f"{script_path.stem}_{timestamp}.mp4"


def _format_eta(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def main():
    parser = argparse.ArgumentParser(
        description='Convert a text script into an edited documentary-style video.'
    )
    parser.add_argument('script_file', help='Path to input script (text file)')
    parser.add_argument('--demo', action='store_true',
                        help='Placeholder visuals instead of stock footage (no API keys needed)')
    parser.add_argument('--keep-temp', action='store_true',
                        help='Keep intermediate/temp files after export')
    parser.add_argument('--scene-duration', type=float, default=4.0, metavar='SEC',
                        help='Target scene (caption) length in seconds (default: 4; '
                             '6-8 recommended for 30+ minute videos)')
    parser.add_argument('--max-shot', type=float, default=4.5, metavar='SEC',
                        help='Longest a single clip stays on screen; longer scenes are '
                             'built from multiple clips (default: 4.5; 0 disables)')
    parser.add_argument('--transition', choices=('varied', 'fade', 'cut'), default='varied',
                        help='varied: planned mix of cuts, dips to black/white and '
                             'punch-ins per boundary (default); fade: uniform edge '
                             'fades; cut: hard cuts only')
    parser.add_argument('--no-motion', action='store_true',
                        help='Disable Ken Burns motion (faster renders)')
    parser.add_argument('--no-grade', action='store_true',
                        help='Disable the contrast grade + vignette look')
    parser.add_argument('--no-callout', action='store_true',
                        help='Disable on-screen entity callouts (place/person/number tags)')
    parser.add_argument('--no-vision', action='store_true',
                        help='Skip CLIP visual verification of stock candidates')
    parser.add_argument('--no-cards', action='store_true',
                        help='Disable full-screen statement cards on power lines')
    parser.add_argument('--no-sfx', action='store_true',
                        help='Disable whoosh/impact/riser sound design')
    parser.add_argument('--music', type=Path, default=None, metavar='AUDIO',
                        help='Background music file — looped, ducked under narration, '
                             'faded in/out')
    parser.add_argument('--voiceover', type=Path, default=None, metavar='AUDIO',
                        help='Narration track — muxed at full volume; scene pacing '
                             'auto-calibrates to its length unless --wps is set')
    parser.add_argument('--wps', type=float, default=None, metavar='RATE',
                        help='Speaking pace in words/second (default: 2.5, or derived '
                             'from --voiceover)')
    parser.add_argument('--export-plan', type=Path, default=None, metavar='JSON',
                        help='Write the scene plan (text, timing, keywords) to JSON '
                             'and exit — hand-edit keywords, then re-run with --use-plan')
    parser.add_argument('--use-plan', type=Path, default=None, metavar='JSON',
                        help='Override scene keywords from an edited plan file')
    parser.add_argument('--audit', action='store_true',
                        help='Select clips for every scene and save each pick\'s '
                             'thumbnail + a match report to output/audit/ WITHOUT '
                             'rendering — verify picks by eye first')
    parser.add_argument('--limit', type=int, default=None, metavar='N',
                        help='Render only the first N scenes — for a quick style proof '
                             'before committing to a long full render')
    parser.add_argument('--progress-every', type=int, default=10, metavar='N',
                        help='Print a progress line every N completed scenes (default: 10)')
    parser.add_argument('--workers', type=int,
                        default=max(1, min(3, (os.cpu_count() or 2) - 1)),
                        metavar='N',
                        help='Parallel scene render processes (default: cores-1, max 3). '
                             'Searches stay single-threaded for rate-limit safety.')

    args = parser.parse_args()

    script_path = Path(args.script_file)
    if not script_path.exists():
        print(f"Error: Script file not found: {script_path}")
        sys.exit(1)
    for f in (args.music, args.voiceover, args.use_plan):
        if f is not None and not f.exists():
            print(f"Error: File not found: {f}")
            sys.exit(1)

    try:
        with PipelineLock(TEMP_DIR / "pipeline.lock"):
            _run_pipeline(args, script_path)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


def _run_pipeline(args, script_path: Path):
    # Steps 1-2: scenes + keywords
    print(f"Loading script: {script_path}")
    script = load_script(str(script_path))

    scene_word_times = None
    if args.voiceover is not None:
        # Align to the narrator's actual voice. Preferred: Whisper word
        # timestamps (every word's true spoken time). Fallback: pause
        # detection from the waveform.
        from nltk.tokenize import sent_tokenize
        from scene_parser import split_into_scenes_aligned
        sentences = sent_tokenize(script.strip())
        tokens = script.split()
        word_times = None
        try:
            from align import align_words
            print(f"Aligning {len(tokens)} script words to the voiceover (Whisper)...")
            word_times = align_words(tokens, str(args.voiceover))
        except Exception as exc:
            print(f"  Whisper alignment unavailable ({str(exc)[:100]}) — "
                  f"falling back to pause detection")
            from align import align_sentences
            times = align_sentences([len(s.split()) for s in sentences],
                                    str(args.voiceover))

        if word_times is not None:
            # Sentence spans from their words' true times
            times = []
            ptr = 0
            for s in sentences:
                n = len(s.split())
                times.append((word_times[ptr][0], word_times[ptr + n - 1][1]))
                ptr += n
            # Video timeline starts at 0 even if narration starts later
            times[0] = (0.0, times[0][1])

        pace = len(tokens) / max(times[-1][1], 1e-6)
        print(f"  voiceover: {times[-1][1]:.1f}s | pace {pace:.2f} words/sec")
        scenes = split_into_scenes_aligned(script, times,
                                           target_duration=args.scene_duration)

        if word_times is not None:
            # Per-scene word times, relative to each scene's start — these
            # drive the karaoke highlight to the narrator's exact voice
            scene_word_times = []
            ptr = 0
            for sc in scenes:
                n = len(sc.text.split())
                span = word_times[ptr:ptr + n]
                scene_word_times.append(
                    [(max(0.0, s - sc.start_time), max(0.05, e - sc.start_time))
                     for s, e in span]
                )
                ptr += n
    else:
        pace = args.wps if args.wps is not None else 2.5
        print("Splitting into scenes and extracting keywords...")
        scenes = split_into_scenes(script, target_duration=args.scene_duration,
                                   words_per_second=pace)
    print_scenes(scenes, limit=10 if len(scenes) > 40 else None)

    # Plan workflow: export keywords for hand-editing, or apply edits
    if args.export_plan is not None:
        plan = [
            {"scene": i + 1, "start": round(s.start_time, 2), "end": round(s.end_time, 2),
             "text": s.text, "keywords": s.keywords}
            for i, s in enumerate(scenes)
        ]
        args.export_plan.write_text(json.dumps(plan, indent=2))
        print(f"Scene plan written to {args.export_plan} — edit the keywords, then "
              f"re-run with --use-plan {args.export_plan}")
        return
    if args.use_plan is not None:
        plan = json.loads(args.use_plan.read_text())
        if len(plan) < len(scenes):
            raise RuntimeError(
                f"Plan has {len(plan)} scenes but the script splits into "
                f"{len(scenes)} — re-export the plan after script/settings changes."
            )
        for entry, scene in zip(plan, scenes):
            scene.keywords = list(entry["keywords"])
        print(f"Applied keyword overrides from {args.use_plan}")

    # Style-proof: render only the first N scenes
    if args.limit is not None and args.limit < len(scenes):
        scenes = scenes[:args.limit]
        print(f"--limit {args.limit}: rendering only the first {len(scenes)} scenes "
              f"({scenes[-1].end_time:.1f}s) as a proof.")

    # Documentary treatment plans (deterministic — resume-safe)
    from effects import (SUBSHOT_OVERLAP, plan_transitions, plan_motion,
                         scene_edge_styles)
    from elements import plan_cards
    boundaries = plan_transitions(len(scenes)) if args.transition == 'varied' else None
    motion_plan = None if args.no_motion else plan_motion(len(scenes))
    card_flags = [False] * len(scenes) if args.no_cards else plan_cards(scenes)
    if any(card_flags):
        idxs = [i + 1 for i, f in enumerate(card_flags) if f]
        print(f"Statement cards on scenes: {idxs}")

    # Keywords are part of the run identity: editing the shot list must
    # invalidate old rendered scenes so they re-render with new footage.
    keywords_sig = "|".join(",".join(s.keywords) for s in scenes)
    settings = (f"{args.scene_duration}|{pace:.3f}|{args.max_shot}|{args.transition}|"
                f"motion={not args.no_motion}|grade={not args.no_grade}|"
                f"callout={not args.no_callout}|cards={not args.no_cards}|v6-retention|"
                f"kw={hashlib.sha256(keywords_sig.encode()).hexdigest()[:12]}")
    run_id = fingerprint(script, len(scenes), settings)
    checkpoint = Checkpoint(TEMP_DIR / "progress.json", run_id)
    already_done = sum(
        1 for i in range(len(scenes)) if checkpoint.done_path(i) is not None
    )
    if already_done:
        print(f"Resuming: {already_done}/{len(scenes)} scenes already complete.\n")

    from editor import concat_scene_files, mux_audio
    from clip_finder import ClipSearchError, check_api_access, find_clip_for_scene

    if args.demo:
        print("Demo mode: generating placeholder visuals (Steps 3-4 skipped)...")
    else:
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
    scenes_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    scene_paths: dict[int, Path] = {}
    pending: list[int] = []
    for i in range(len(scenes)):
        existing = checkpoint.done_path(i)
        if existing is not None:
            scene_paths[i] = existing
        else:
            pending.append(i)

    def make_job(i: int) -> dict:
        """Select clips for scene i (search is main-process only) and build
        the worker job dict."""
        scene = scenes[i]
        if boundaries is not None:
            in_style, out_style = scene_edge_styles(i, len(scenes), boundaries)
        else:
            style = "black" if args.transition == "fade" else "cut"
            in_style, out_style = style, style
        if i == 0:
            in_style = "black"
        if i == len(scenes) - 1:
            out_style = "black"

        shots = None
        if not args.demo and not card_flags[i]:
            if args.max_shot > 0:
                n_shots = min(MAX_SHOTS_PER_SCENE,
                              max(1, math.ceil(scene.duration / args.max_shot)))
            else:
                n_shots = 1
            # Sub-shots are cut SUBSHOT_OVERLAP longer so the crossfade
            # overlap leaves the assembled scene at exactly scene.duration.
            shot_duration = (scene.duration + SUBSHOT_OVERLAP * (n_shots - 1)) / n_shots
            shots = []
            for _ in range(n_shots):
                candidate = find_clip_for_scene(scene, i, used_ids,
                                                use_vision=not args.no_vision)
                if candidate is None:
                    raise ClipSearchError(
                        f"Scene {i + 1}: no stock results on Pexels or Pixabay for "
                        f"keywords {scene.keywords} — try broader keywords or edit "
                        f"them via --export-plan/--use-plan"
                    )
                shots.append({"candidate": asdict(candidate), "duration": shot_duration})

        return {
            "index": i, "n_scenes": len(scenes),
            "text": scene.text, "keywords": scene.keywords,
            "start": scene.start_time, "end": scene.end_time,
            "shots": shots,
            "in_style": in_style, "out_style": out_style,
            "motion": (motion_plan[i]
                       if (motion_plan is not None and in_style not in ("punch", "zoompunch"))
                       else None),
            "grade": not args.no_grade,
            "callout": not args.no_callout,
            "card": card_flags[i],
            "word_times": scene_word_times[i] if scene_word_times else None,
            "raw_dir": str(raw_dir),
            "scene_file": str(scenes_dir / f"scene_{i + 1:04d}.mp4"),
        }

    if args.audit:
        import requests as _rq
        audit_dir = Path("output/audit")
        if audit_dir.exists():
            shutil.rmtree(audit_dir)
        audit_dir.mkdir(parents=True, exist_ok=True)
        report = []
        for i in range(len(scenes)):
            job = make_job(i)
            if job.get("card") or not job.get("shots"):
                continue
            for n, shot in enumerate(job["shots"]):
                c = shot["candidate"]
                entry = {"scene": i + 1, "shot": n + 1,
                         "text": scenes[i].text[:90],
                         "query": c["query"], "clip": f"{c['source']}:{c['video_id']}",
                         "sim": round(c["sim"], 3), "desc": c["desc"][:100]}
                report.append(entry)
                if c.get("thumb"):
                    try:
                        img = _rq.get(c["thumb"], timeout=15).content
                        (audit_dir / f"scene_{i+1:04d}_{n+1}_sim{c['sim']:.2f}.jpg"
                         ).write_bytes(img)
                    except Exception:
                        pass
        (audit_dir / "picks_report.json").write_text(json.dumps(report, indent=2))
        strong = sum(1 for r in report if r["sim"] >= 0.24)
        ok = sum(1 for r in report if 0.19 <= r["sim"] < 0.24)
        weak = sum(1 for r in report if r["sim"] < 0.19)
        print(f"\nAUDIT: {len(report)} picks | strong(>=0.24): {strong} | "
              f"ok(0.19-0.24): {ok} | weak(<0.19): {weak}")
        for r in report:
            if r["sim"] < 0.19:
                print(f"  WEAK scene {r['scene']}: sim={r['sim']} q='{r['query']}' -> {r['desc'][:60]}")
        print(f"Thumbnails + picks_report.json in {audit_dir}/")
        return

    from scene_builder import render_scene_job

    run_start = time.monotonic()
    rendered_this_run = 0
    max_in_flight = max(2, args.workers * 2)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        in_flight = {}
        queue = iter(pending)
        exhausted = False
        try:
            while in_flight or not exhausted:
                while not exhausted and len(in_flight) < max_in_flight:
                    try:
                        i = next(queue)
                    except StopIteration:
                        exhausted = True
                        break
                    job = make_job(i)
                    in_flight[pool.submit(render_scene_job, job)] = (i, job["scene_file"])

                if not in_flight:
                    break
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    i, scene_file = in_flight.pop(fut)
                    fut.result()  # re-raises worker failures
                    scene_paths[i] = Path(scene_file)
                    checkpoint.mark_done(i, Path(scene_file), used_ids)
                    rendered_this_run += 1

                    if (rendered_this_run % args.progress_every == 0
                            or rendered_this_run == len(pending)):
                        elapsed = time.monotonic() - run_start
                        remaining = len(pending) - rendered_this_run
                        per_scene = elapsed / rendered_this_run
                        print(f"Scene {len(scene_paths)}/{len(scenes)} complete "
                              f"| {per_scene:.1f}s/scene ({args.workers} workers) "
                              f"| ETA {_format_eta(remaining * per_scene)}")
        except Exception:
            for fut in in_flight:
                fut.cancel()
            raise

    scene_files = [scene_paths[i] for i in range(len(scenes))]

    # Steps 5+7: fast final stitch (no re-encode), then audio
    output_path = build_output_path(script_path)
    print(f"\nAssembling {len(scene_files)} scenes with ffmpeg concat -> {output_path}")
    concat_scene_files(scene_files, output_path)

    total_duration = sum(s.duration for s in scenes)
    sfx_track = None
    if not args.no_sfx:
        from sfx import build_sfx_track, plan_sfx_events
        events = plan_sfx_events(scenes, boundaries, card_flags)
        sfx_track = build_sfx_track(events, total_duration,
                                    Path("output") / ".sfx_track.wav")
        if sfx_track is not None:
            print(f"Sound design: {len(events)} audio accents "
                  f"(whooshes/impacts/risers) placed at scene boundaries")
    if args.music or args.voiceover or sfx_track:
        print("Muxing audio (voiceover full volume, music ducked, sfx accents)...")
        mux_audio(output_path, total_duration,
                  music=args.music, voiceover=args.voiceover, sfx=sfx_track)
        if sfx_track is not None:
            sfx_track.unlink(missing_ok=True)

    # Clean up scene clips + checkpoint after a successful export
    if not args.keep_temp and TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
        print("Cleaned up temp files (use --keep-temp to keep them).")
    elif args.keep_temp and TEMP_DIR.exists():
        print(f"Temp files kept in {TEMP_DIR}/")

    print(f"\nDone: {output_path}")


if __name__ == '__main__':
    main()
