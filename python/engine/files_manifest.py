"""
Files manifest — append-only insertion-order log of input files.

Lives at ~/.basevault/files-manifest.json (override via env var
BASEVAULT_FILES_MANIFEST). Each entry is

    {"file_path": str, "content_hash": str (sha256 hex), "added_at": str (ISO-Z)}

Insertion order = the order in which files were FIRST observed across
runs. Stable under appends: a new file appended at run N+1 keeps every
prior entry's position. This is the load-bearing property the LLM
prompt cache relies on for "add a new file → only that file's stage
calls bust" behavior — entity batching, splitter ordering, and any
other input-order-dependent stage reads from this manifest instead of
re-sorting `os.listdir()` output per run.

Manifest semantics:
- Keyed by `file_path`. A path appears at most once.
- First sighting: append `{path, content_hash=sha256(file), added_at=now}`.
- Re-sighting with same content_hash: no-op.
- Re-sighting with different content_hash: update `content_hash` in
  place; `added_at` and position stay frozen. Content change does not
  re-order; it only invalidates downstream caches keyed on the file's
  content hash.
- File deletion: entries are NEVER removed by the writer. (Stale
  entries are harmless — readers iterate by file_path lookup, so a
  path that no longer exists on disk just doesn't get queried.)

Concurrency: an exclusive `fcntl.flock` on `<manifest>.lock` guards the
read-merge-write sequence across processes. Sweep harnesses run many
pipeline subprocesses against the shared manifest; without the lock,
two of them race on `<manifest>.tmp` and `os.replace`, producing
either lost updates or `FileNotFoundError` from a vanished tmp file.
A module-level `threading.Lock` covers the in-process case for
callers that share an interpreter (tests, the wizard, …).
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path


# ── Path resolution ───────────────────────────────────────────────────────────

def _default_manifest_path() -> Path:
    override = os.environ.get("BASEVAULT_FILES_MANIFEST")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".basevault" / "files-manifest.json"


# Module-level lock guards the in-process critical section. Cross-
# process coordination is done via `_file_lock()` (fcntl.flock on a
# sibling .lock file) inside `update_manifest`. Two concurrent
# subprocesses serialize on the flock; neither sees a half-merged
# manifest, neither loses an entry the other appended.
_lock = threading.Lock()


@contextlib.contextmanager
def _file_lock(lock_path: Path):
    """Cross-process exclusive lock on `lock_path`.

    Acquired for the duration of the `with` block. The lock file is
    created if missing and is never deleted — `fcntl.flock` keys on
    the open file descriptor, so the same inode is reused across
    processes. Posix-only; on Windows we'd swap in `msvcrt.locking`,
    but BaseVault is mac-first and the sweep harness only runs on
    macOS / Linux today.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # Open in append-create mode so two processes don't race on an
    # initial truncate.
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _lock_path_for(manifest_path: Path) -> Path:
    """Sibling `.lock` file, kept distinct from the manifest itself so
    `os.replace(tmp, manifest)` doesn't disturb an open lock fd."""
    return manifest_path.with_suffix(manifest_path.suffix + ".lock")


@dataclass(frozen=True)
class ManifestEntry:
    file_path: str
    content_hash: str
    added_at: str


from engine.common.dates import now_iso_z as _now_iso_z  # noqa: E402


def _hash_file(path: Path, _chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(_chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def hash_file(path: str | Path) -> str:
    """Public sha256 helper. Reads the file in 1 MB chunks so 100 MB
    inputs don't balloon memory."""
    return _hash_file(Path(path))


# ── Read / write ──────────────────────────────────────────────────────────────

def load_manifest(path: Path | None = None) -> list[ManifestEntry]:
    p = path or _default_manifest_path()
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError, OSError):
        return []
    if not isinstance(raw, list):
        return []
    out: list[ManifestEntry] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        fp = entry.get("file_path")
        ch = entry.get("content_hash")
        ad = entry.get("added_at")
        if not (isinstance(fp, str) and isinstance(ch, str) and isinstance(ad, str)):
            continue
        out.append(ManifestEntry(file_path=fp, content_hash=ch, added_at=ad))
    return out


def _atomic_write(path: Path, payload: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def update_manifest(
    discovered: list[tuple[str, str]],
    path: Path | None = None,
) -> list[ManifestEntry]:
    """Reconcile `discovered` (list of (file_path, content_hash)) against
    the on-disk manifest. New paths append at the end; existing paths
    update content_hash if changed (added_at + position stay frozen).
    Returns the full ordered manifest after the merge.

    Pure for repeated calls: passing the same discovered list twice
    yields the same manifest both times.

    Concurrency: the read-merge-write sequence is serialized by
    `_file_lock` (`fcntl.flock` on a sibling `.lock` file) so
    concurrent runner subprocesses don't lose appends or race on
    `<manifest>.tmp`.
    """
    p = path or _default_manifest_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with _lock, _file_lock(_lock_path_for(p)):
        entries = load_manifest(p)
        by_path = {e.file_path: idx for idx, e in enumerate(entries)}
        mutated = False
        now = _now_iso_z()
        for file_path, content_hash in discovered:
            idx = by_path.get(file_path)
            if idx is None:
                entries.append(ManifestEntry(
                    file_path=file_path,
                    content_hash=content_hash,
                    added_at=now,
                ))
                by_path[file_path] = len(entries) - 1
                mutated = True
            elif entries[idx].content_hash != content_hash:
                # Content drift: update hash but preserve position +
                # added_at. A future "what's new" surface uses
                # added_at as first-sighting timestamp; updating it on
                # every edit would defeat that.
                old = entries[idx]
                entries[idx] = ManifestEntry(
                    file_path=old.file_path,
                    content_hash=content_hash,
                    added_at=old.added_at,
                )
                mutated = True
        if mutated:
            _atomic_write(p, [
                {
                    "file_path": e.file_path,
                    "content_hash": e.content_hash,
                    "added_at": e.added_at,
                }
                for e in entries
            ])
        return entries


def discover_and_update(
    paths: list[Path],
    base_dir: Path | None = None,
    path_strategy: str = "absolute",
    manifest_path: Path | None = None,
) -> list[ManifestEntry]:
    """Compute content hashes for `paths` and merge into the manifest.
    Returns the resulting full manifest (ordered).

    `path_strategy`:
      "absolute" — record str(p.resolve()) as the manifest key.
      "relative" — record str(p.relative_to(base_dir)) as the key.
        Useful when the same input dir gets moved across machines and
        you want manifest portability.
    """
    discovered: list[tuple[str, str]] = []
    for p in paths:
        if path_strategy == "relative" and base_dir is not None:
            try:
                key = str(p.relative_to(base_dir))
            except ValueError:
                key = str(p.resolve())
        else:
            key = str(p.resolve())
        try:
            ch = _hash_file(p)
        except OSError:
            continue
        discovered.append((key, ch))
    return update_manifest(discovered, path=manifest_path)


def position_index(entries: list[ManifestEntry]) -> dict[str, int]:
    """`{file_path: insertion_position}` lookup.

    Entities batching uses this to assign each canonical entity its
    earliest-source manifest position, which becomes the cache-stable
    sort key.
    """
    return {e.file_path: idx for idx, e in enumerate(entries)}
