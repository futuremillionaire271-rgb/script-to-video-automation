"""
Resumable pipeline checkpointing.

Progress is written to temp/progress.json after every completed scene, so an
interrupted run picks up from the last finished scene instead of starting
over. The checkpoint is invalidated (fresh start) if the script text or
scene settings change between runs.
"""

import hashlib
import json
import os
from pathlib import Path


class PipelineLock:
    """
    Guard against two pipeline runs sharing temp/ at once — the second run
    would race the first's per-scene files and post-export cleanup.
    Stale locks (dead PID) are reclaimed automatically.
    """

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        if self.path.exists():
            try:
                other_pid = int(self.path.read_text().strip())
                os.kill(other_pid, 0)  # raises if PID is gone
                raise RuntimeError(
                    f"Another pipeline run (PID {other_pid}) is already using this "
                    f"project directory. Wait for it or stop it, then re-run. "
                    f"(Stale lock? Delete {self.path})"
                )
            except (ValueError, ProcessLookupError, PermissionError):
                pass  # stale or unreadable lock — take it over
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(str(os.getpid()))
        return self

    def __exit__(self, *exc):
        if self.path.exists():
            self.path.unlink(missing_ok=True)
        return False


def fingerprint(script_text: str, scene_count: int, settings: str) -> str:
    """
    Identity of a run: same script + same render settings. Any change to
    the settings string invalidates old checkpoints (scenes rendered with
    different motion/transition/pacing settings must not be mixed).
    """
    h = hashlib.sha256()
    h.update(script_text.encode("utf-8"))
    h.update(f"|{scene_count}|{settings}".encode())
    return h.hexdigest()[:16]


class Checkpoint:
    def __init__(self, path: Path, run_fingerprint: str):
        self.path = path
        self.fingerprint = run_fingerprint
        self.completed: dict[str, str] = {}   # scene index (str) -> clip path
        self.used_clips: list[str] = []       # "source:id" of every clip used

        if path.exists():
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                data = {}
            if data.get("fingerprint") == run_fingerprint:
                self.completed = data.get("completed", {})
                self.used_clips = data.get("used_clips", [])
            else:
                print("  Checkpoint exists but script/settings changed — starting fresh.")

    def done_path(self, index: int) -> Path | None:
        """Return the rendered clip path for a completed scene, if still on disk."""
        p = self.completed.get(str(index))
        if p and Path(p).exists():
            return Path(p)
        return None

    def mark_done(self, index: int, clip_path: Path, used_clips: set[str]) -> None:
        """Record a completed scene and persist atomically."""
        self.completed[str(index)] = str(clip_path)
        self.used_clips = sorted(used_clips)
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "fingerprint": self.fingerprint,
            "completed": self.completed,
            "used_clips": self.used_clips,
        }, indent=2))
        tmp.replace(self.path)
