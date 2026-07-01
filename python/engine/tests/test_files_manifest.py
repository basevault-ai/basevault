"""
Tests for the append-only files manifest.

The manifest is the load-bearing source of stable input ordering for
the LLM prompt cache: a re-run on identical inputs must produce
identical batch packing, and ADDING a new file must keep every prior
file's manifest position frozen.
"""
from __future__ import annotations

import json

import pytest

from engine import files_manifest


@pytest.fixture(autouse=True)
def _isolated_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("BASEVAULT_FILES_MANIFEST", str(tmp_path / "manifest.json"))
    yield


def _entries() -> list[files_manifest.ManifestEntry]:
    return files_manifest.load_manifest()


def test_load_empty_manifest_returns_empty_list(tmp_path):
    assert _entries() == []


def test_first_run_appends_entries_in_input_order():
    out = files_manifest.update_manifest([
        ("a.txt", "h_a"),
        ("b.txt", "h_b"),
        ("c.txt", "h_c"),
    ])
    assert [e.file_path for e in out] == ["a.txt", "b.txt", "c.txt"]
    pos = files_manifest.position_index(out)
    assert pos == {"a.txt": 0, "b.txt": 1, "c.txt": 2}


def test_re_running_with_same_inputs_is_a_noop():
    files_manifest.update_manifest([("a.txt", "h_a"), ("b.txt", "h_b")])
    out2 = files_manifest.update_manifest([("a.txt", "h_a"), ("b.txt", "h_b")])
    assert [e.file_path for e in out2] == ["a.txt", "b.txt"]
    # added_at on the existing entries must not change.
    after_added_ats = [e.added_at for e in out2]
    out3 = files_manifest.update_manifest([("a.txt", "h_a"), ("b.txt", "h_b")])
    assert [e.added_at for e in out3] == after_added_ats


def test_adding_a_new_file_appends_at_the_end_and_preserves_positions():
    files_manifest.update_manifest([
        ("a.txt", "h_a"),
        ("b.txt", "h_b"),
        ("c.txt", "h_c"),
    ])
    pos_before = files_manifest.position_index(_entries())
    out_after = files_manifest.update_manifest([
        ("a.txt", "h_a"),
        ("b.txt", "h_b"),
        ("c.txt", "h_c"),
        ("d.txt", "h_d"),
    ])
    pos_after = files_manifest.position_index(out_after)
    assert pos_after["a.txt"] == pos_before["a.txt"] == 0
    assert pos_after["b.txt"] == pos_before["b.txt"] == 1
    assert pos_after["c.txt"] == pos_before["c.txt"] == 2
    assert pos_after["d.txt"] == 3


def test_content_change_updates_hash_in_place_without_reordering():
    """Editing a file's content rewrites its content_hash but leaves
    its manifest position and added_at frozen — load-bearing for the
    'what's new' surface that uses added_at as first-sighting time."""
    out_initial = files_manifest.update_manifest([
        ("a.txt", "h_a_v1"),
        ("b.txt", "h_b"),
    ])
    a_added_at = out_initial[0].added_at
    out_after = files_manifest.update_manifest([
        ("a.txt", "h_a_v2"),
        ("b.txt", "h_b"),
    ])
    assert out_after[0].file_path == "a.txt"
    assert out_after[0].content_hash == "h_a_v2"
    assert out_after[0].added_at == a_added_at
    # Position unchanged.
    assert files_manifest.position_index(out_after)["a.txt"] == 0


def test_manifest_is_persistent_across_processes(tmp_path):
    """Atomic write means a fresh `load_manifest()` (after the writer
    completes) sees the new entries — same as a sibling Python
    process would in a sweep harness."""
    files_manifest.update_manifest([("a.txt", "h_a")])
    # Simulate a fresh process by clearing module state and re-loading.
    out = files_manifest.load_manifest()
    assert len(out) == 1
    assert out[0].file_path == "a.txt"
    # Round-trip JSON is well-formed.
    raw = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert isinstance(raw, list) and raw[0]["file_path"] == "a.txt"


def test_corrupt_manifest_returns_empty_list_not_crash(tmp_path):
    (tmp_path / "manifest.json").write_text("garbage", encoding="utf-8")
    assert files_manifest.load_manifest() == []


def test_position_index_handles_empty_manifest():
    assert files_manifest.position_index([]) == {}


def test_hash_file_round_trip(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"hello world")
    h1 = files_manifest.hash_file(p)
    h2 = files_manifest.hash_file(p)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


# ── Cross-process locking regression ────────────────────────────────────────


def _concurrent_appender(args: tuple[str, str, str]) -> None:
    """Worker process: append a single (file_path, hash) to the
    shared manifest. Re-imports the module since `multiprocessing`
    spawn gives the child a fresh interpreter."""
    from pathlib import Path as _Path
    manifest_path_str, file_path, content_hash = args
    from engine import files_manifest as fm_child
    fm_child.update_manifest(
        [(file_path, content_hash)], path=_Path(manifest_path_str),
    )


def test_concurrent_subprocess_appends_no_lost_updates(tmp_path):
    """Regression: in PR #66 sweep, two runner subprocesses raced on
    `_atomic_write` and the second crashed with `FileNotFoundError:
    <manifest>.tmp`. The fix is an `fcntl.flock` on a sibling
    `.lock` file. Every concurrent appender must land in the final
    manifest, and the JSON must parse cleanly."""
    import multiprocessing
    manifest_path = tmp_path / "concurrent-manifest.json"
    n_workers = 8
    jobs = [
        (str(manifest_path), f"file_{i}.txt", f"hash_{i}")
        for i in range(n_workers)
    ]
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        pool.map(_concurrent_appender, jobs)

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(raw, list)
    seen = {entry["file_path"] for entry in raw}
    expected = {f"file_{i}.txt" for i in range(n_workers)}
    assert expected.issubset(seen), (
        f"missing {expected - seen} from concurrent writes — "
        f"some subprocess lost its append"
    )
    paths = [entry["file_path"] for entry in raw]
    assert len(paths) == len(set(paths)), "duplicate file_path keys"


def test_lock_file_lives_alongside_manifest(tmp_path):
    """Sibling `.lock` file is created on first write and stays put
    so concurrent processes serialize on the same inode across the
    lifetime of the manifest."""
    manifest_path = tmp_path / "manifest.json"
    files_manifest.update_manifest([("a.txt", "h1")], path=manifest_path)
    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
    assert lock_path.exists()
    files_manifest.update_manifest([("b.txt", "h1")], path=manifest_path)
    assert lock_path.exists()
