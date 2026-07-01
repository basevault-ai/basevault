"""Tests for scripts/code_gates.py — the code-correctness gates.

Gates under test:
  (a) listen()-in-useEffect cancelled guard

A violating fixture trips it, known-good fixtures stay clean, and the real
tree is clean. The guard module is loaded via sys.path.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
import code_gates as guards  # noqa: E402  (sys.path injected just above)


# --- Gate (a): listen()-in-useEffect cancelled guard ------------------------

_BAD_LISTEN = '''\
function C() {
  useEffect(() => {
    let unlisten;
    listen("ask-event", (e) => append(e.payload)).then((u) => {
      unlisten = u;
    });
    return () => unlisten?.();
  }, []);
}
'''

_GOOD_CANCELLED = '''\
function C() {
  useEffect(() => {
    let unlisten;
    let cancelled = false;
    listen("ask-event", (e) => append(e.payload)).then((u) => {
      if (cancelled) { u(); return; }
      unlisten = u;
    });
    return () => { cancelled = true; if (unlisten) unlisten(); };
  }, []);
}
'''

_GOOD_LISTEN_MARKER = '''\
function C() {
  useEffect(() => {
    // ci-allow:listen-guard — idempotent boolean setState, benign
    let unlisten;
    listen("open-settings", () => setOpen(true)).then((u) => {
      unlisten = u;
    });
    return () => unlisten?.();
  }, []);
}
'''

# listen() in a useCallback / event handler is not subject to the
# StrictMode mount->cleanup->remount double-invoke — out of the family.
_LISTEN_OUTSIDE_EFFECT = '''\
function C() {
  const download = useCallback(async () => {
    const unlisten = await listen("progress", (e) => setMsg(e.payload));
    await invoke("download");
    unlisten?.();
  }, []);
}
'''


def _run_gate_a(tmp_path, monkeypatch, contents: str, name: str = "Comp.jsx"):
    src = tmp_path / "src"
    src.mkdir()
    (src / name).write_text(contents, encoding="utf-8")
    monkeypatch.setattr(guards, "SRC_DIR", src)
    monkeypatch.setattr(guards, "REPO_ROOT", tmp_path)
    return guards._gate_a_violations()


def test_gate_a_trips_on_uncancelled_listen_in_effect(tmp_path, monkeypatch):
    assert _run_gate_a(tmp_path, monkeypatch, _BAD_LISTEN)


def test_gate_a_clean_with_cancelled_guard(tmp_path, monkeypatch):
    assert _run_gate_a(tmp_path, monkeypatch, _GOOD_CANCELLED) == []


def test_gate_a_clean_with_inline_marker(tmp_path, monkeypatch):
    assert _run_gate_a(tmp_path, monkeypatch, _GOOD_LISTEN_MARKER) == []


def test_gate_a_ignores_listen_outside_useeffect(tmp_path, monkeypatch):
    assert _run_gate_a(tmp_path, monkeypatch, _LISTEN_OUTSIDE_EFFECT) == []


def test_gate_a_skips_test_files(tmp_path, monkeypatch):
    assert _run_gate_a(
        tmp_path, monkeypatch, _BAD_LISTEN, name="Comp.test.jsx"
    ) == []


def test_real_tree_is_clean():
    assert guards._gate_a_violations() == []
