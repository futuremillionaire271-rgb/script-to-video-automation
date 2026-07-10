"""
Resumable pipeline checkpointing.

Progress is written to temp/progress.json after every completed scene, so an
interrupted run picks up from the last finished scene instead of starting
over. The checkpoint is invalidated (fresh start) if the script text or
scene settings change between runs.
"""

import hashlib
import json
from pathlib import Path


def fingerprint(script_text: str, scene_count: int, scene_duration: float) -> str:
    """Identity of a run: same script + same splitting settings."""
    h = hashlib.sha256()
    h.update(script_text.encode("utf-8"))
    h.update(f"|{scene_count}|{scene_duration}".encode())
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
