"""Tests for scripts/trust_gates.py — the trust-surface / supply-chain gates.

Gates under test:
  (a) no banned cloud-provider token outside testing/
  (b) production tree (engine + kernel + ...) never imports testing/
  (c) the pipeline (engine) never imports the TEE-reaching set (#909)
  (d) no dynamic-import tokens in engine or kernel
  (e) no tracked .env

Each gate: a violating fixture trips it, a known-good fixture stays clean,
and the real-tree run is clean (gate c has no baseline exemptions — the
engine reaches the enclave only through the kernel's attested provider).

This file is under python/engine/tests, so gate (d) scans it: it must never
contain the literal banned tokens. The guard module is loaded via sys.path
(not the dynamic-import API), and the gate-(d) fixtures build the forbidden
tokens by concatenation so the substrings never appear here.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
import trust_gates as guards  # noqa: E402  (sys.path injected just above)


# --- Gate (a): no banned cloud-provider token outside testing/ --------------

def _run_gate_a(tmp_path, monkeypatch, contents: str, rel: str):
    """Write ``contents`` to ``tmp_path/<rel>`` and scan with gate (a)
    rooted at ``tmp_path``. ``rel`` may place the file under
    ``python/testing/`` to exercise the sanctioned-home exemption."""
    target = tmp_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(contents, encoding="utf-8")
    monkeypatch.setattr(guards, "APP_DIR", tmp_path)
    monkeypatch.setattr(guards, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(guards, "TESTING_DIR", tmp_path / "python" / "testing")
    # SELF_PATH must point outside the tmp tree so the fixture isn't
    # mistaken for the guard's own (legitimately token-bearing) source.
    monkeypatch.setattr(guards, "SELF_PATH", tmp_path / "__not_self__.py")
    return guards._gate_a_violations()


def test_gate_a_trips_on_banned_token_in_engine(tmp_path, monkeypatch):
    body = "PROVIDER = 'fire" + "works'\n"  # split so this test file stays clean
    assert _run_gate_a(tmp_path, monkeypatch, body, "python/engine/llm.py")


def test_gate_a_exempts_testing_tree(tmp_path, monkeypatch):
    body = "PROVIDER = 'fire" + "works'\n"
    assert _run_gate_a(
        tmp_path, monkeypatch, body, "python/testing/common/provider.py"
    ) == []


# --- Gate (b): production tree (engine + kernel) never imports testing/ ------

def _run_gate_b(tmp_path, monkeypatch, contents: str, rel: str):
    target = tmp_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(contents, encoding="utf-8")
    monkeypatch.setattr(
        guards, "PRODUCTION_DIRS",
        (tmp_path / "python" / "engine", tmp_path / "python" / "kernel"),
    )
    monkeypatch.setattr(guards, "REPO_ROOT", tmp_path)
    return guards._gate_b_violations()


def test_gate_b_trips_on_testing_import_from_engine(tmp_path, monkeypatch):
    body = "from testing.common.eval_env import Env\n"
    assert _run_gate_b(tmp_path, monkeypatch, body, "python/engine/runner.py")


def test_gate_b_trips_on_testing_import_from_kernel(tmp_path, monkeypatch):
    # The director's requirement: nothing from testing/ in the kernel
    # either. `import testing` (with or without submodule) must trip.
    body = "import testing\n"
    assert _run_gate_b(tmp_path, monkeypatch, body, "python/kernel/thing.py")


def test_gate_b_clean_without_testing_import(tmp_path, monkeypatch):
    body = "from engine.llm import complete\nimport json\n"
    assert _run_gate_b(tmp_path, monkeypatch, body, "python/engine/runner.py") == []


# --- Gate (c): pipeline never imports the TEE-reaching set (#909) ------------

def _run_gate_c(tmp_path, monkeypatch, contents: str, rel: str = "provider.py"):
    engine = tmp_path / "engine"
    (engine / Path(rel).parent).mkdir(parents=True, exist_ok=True)
    (engine / rel).write_text(contents, encoding="utf-8")
    monkeypatch.setattr(guards, "GATE_C_PIPELINE_DIRS", (engine,))
    monkeypatch.setattr(guards, "REPO_ROOT", tmp_path)
    return guards._gate_c_violations()


def test_gate_c_trips_on_toplevel_tinfoil(tmp_path, monkeypatch):
    assert _run_gate_c(tmp_path, monkeypatch, "from tinfoil import TinfoilAI\n")


def test_gate_c_trips_on_lazy_in_function_import(tmp_path, monkeypatch):
    # The AST walk is the whole point: a grep for `^import` would miss this.
    body = "def get():\n    from tinfoil import TinfoilAI\n    return TinfoilAI()\n"
    assert _run_gate_c(tmp_path, monkeypatch, body)


def test_gate_c_trips_on_raw_http(tmp_path, monkeypatch):
    for mod in ("httpx", "requests", "urllib3", "aiohttp"):
        assert _run_gate_c(tmp_path, monkeypatch, f"import {mod}\n"), mod


def test_gate_c_trips_on_urllib_request_but_not_parse(tmp_path, monkeypatch):
    assert _run_gate_c(tmp_path, monkeypatch, "import urllib.request\n")
    assert _run_gate_c(tmp_path, monkeypatch, "import urllib.parse\n") == []


def test_gate_c_trips_on_from_urllib_import_request_and_error(tmp_path, monkeypatch):
    # `from urllib import request, error` — both forbidden names caught.
    v = _run_gate_c(tmp_path, monkeypatch, "from urllib import request, error\n")
    assert len(v) == 2


def test_gate_c_trips_on_attestation_crypto(tmp_path, monkeypatch):
    for mod in ("sigstore", "tuf", "securesystemslib", "cryptography"):
        assert _run_gate_c(tmp_path, monkeypatch, f"import {mod}\n"), mod


def test_gate_c_allows_local_inference(tmp_path, monkeypatch):
    # Director's call: local inference is OK in the engine. mlx / ollama
    # are NOT forbidden — local is in-process / localhost, nothing to attest.
    for mod in ("mlx", "mlx_lm", "mlx_lm.sample_utils", "ollama"):
        assert _run_gate_c(tmp_path, monkeypatch, f"import {mod}\n") == [], mod


def test_gate_c_carveout_exempts_ci_allow_net(tmp_path, monkeypatch):
    body = "import urllib.request  # ci-allow:net - localhost daemon probe\n"
    assert _run_gate_c(tmp_path, monkeypatch, body) == []


def test_gate_c_skips_test_subdir(tmp_path, monkeypatch):
    assert _run_gate_c(
        tmp_path, monkeypatch, "import tinfoil\n", rel="tests/test_provider.py"
    ) == []


# --- Gate (c) matcher unit tests (prefix boundaries + path extraction) -------

def test_gate_c_match_prefix_boundary():
    assert guards._gate_c_match("tinfoil")[0] == "tinfoil"
    assert guards._gate_c_match("tinfoil.client")[0] == "tinfoil"
    # `tinfoilfoo` must NOT match `tinfoil` — it's a different package.
    assert guards._gate_c_match("tinfoilfoo") is None
    # bare `urllib` is stdlib-safe; only request/error submodules are denied.
    assert guards._gate_c_match("urllib") is None
    assert guards._gate_c_match("urllib.parse") is None
    assert guards._gate_c_match("urllib.request")[0] == "urllib.request"
    # `urllib3` is a distinct (forbidden) package, not a urllib submodule.
    assert guards._gate_c_match("urllib3")[0] == "urllib3"
    assert guards._gate_c_match("http")is None
    assert guards._gate_c_match("http.client")[0] == "http.client"


def test_gate_c_imported_paths_relative_import_skipped():
    node = ast.parse("from . import sibling\n").body[0]
    assert guards._gate_c_imported_paths(node) == []
    node2 = ast.parse("from .pkg import thing\n").body[0]
    assert guards._gate_c_imported_paths(node2) == []


def test_gate_c_imported_paths_from_yields_module_and_names():
    node = ast.parse("from urllib import request, error\n").body[0]
    paths = guards._gate_c_imported_paths(node)
    assert "urllib" in paths
    assert "urllib.request" in paths
    assert "urllib.error" in paths


# --- Gate (d): no dynamic-import tokens in engine or kernel ------------------

# The banned tokens, built by concatenation so the literal substrings never
# appear in this file (which gate h scans).
_DUNDER = "__imp" + "ort__"
_ILIB = "import" + "lib"


def _run_gate_d(tmp_path, monkeypatch, contents: str, rel: str = "engine/mod.py"):
    # rel is repo-relative under tmp_path so both engine/ and kernel/ trees
    # can be exercised; GATE_D_DIRS is pointed at both.
    target = tmp_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(contents, encoding="utf-8")
    monkeypatch.setattr(
        guards, "GATE_D_DIRS", (tmp_path / "engine", tmp_path / "kernel")
    )
    monkeypatch.setattr(guards, "REPO_ROOT", tmp_path)
    return guards._gate_d_violations()


def test_gate_d_trips_on_dunder_import(tmp_path, monkeypatch):
    assert _run_gate_d(tmp_path, monkeypatch, f'm = {_DUNDER}("openai")\n')


def test_gate_d_trips_on_ilib_token(tmp_path, monkeypatch):
    assert _run_gate_d(tmp_path, monkeypatch, f"import {_ILIB}\n")


def test_gate_d_is_case_insensitive(tmp_path, monkeypatch):
    assert _run_gate_d(tmp_path, monkeypatch, f"# see {_ILIB.upper()}\n")


def test_gate_d_scans_kernel_too(tmp_path, monkeypatch):
    # Director's call: the ban covers the kernel as well as the engine.
    assert _run_gate_d(
        tmp_path, monkeypatch, f"x = {_DUNDER}('os')\n", rel="kernel/thing.py"
    )


def test_gate_d_scans_tests_under_those_trees(tmp_path, monkeypatch):
    assert _run_gate_d(
        tmp_path, monkeypatch, f'{_DUNDER}("json")\n', rel="engine/tests/test_x.py"
    )


def test_gate_d_clean_with_normal_import(tmp_path, monkeypatch):
    # A bare `import` (the word) is fine — only the two tokens are banned.
    assert _run_gate_d(tmp_path, monkeypatch, "import json\nimport openai\n") == []


# --- Integration: real tree --------------------------------------------------


# --- Gate (e): no tracked .env -----------------------------------------------

def test_gate_e_dotenv_regex_matches_env_files_not_examples():
    m = guards.GATE_E_DOTENV.search
    assert m(".env") and m("oss/.env") and m("foo.env")
    assert not m(".env.example") and not m("config.json")


def test_gate_e_trips_on_tracked_dotenv(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".env").write_text("KEY=secret\n", encoding="utf-8")
    subprocess.run(["git", "add", ".env"], cwd=tmp_path, check=True)
    monkeypatch.setattr(guards, "REPO_ROOT", tmp_path)
    assert guards._gate_e_violations()


def test_gate_e_clean_when_dotenv_untracked(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".env").write_text("KEY=secret\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text(".env\n", encoding="utf-8")
    monkeypatch.setattr(guards, "REPO_ROOT", tmp_path)
    assert guards._gate_e_violations() == []


# --- Integration: real tree --------------------------------------------------

def test_real_tree_is_clean_for_a_b_d_e():
    # Gates a (banned-token), b (testing-imports), d (dynamic-imports), and
    # e (no tracked .env) are clean on the real tree. Gate c is asserted
    # clean directly below (no baseline exemptions).
    assert guards._gate_a_violations() == []
    assert guards._gate_b_violations() == []
    assert guards._gate_d_violations() == []
    assert guards._gate_e_violations() == []


def test_gate_c_is_clean_on_real_tree():
    # The engine reaches the enclave through the kernel's attested provider
    # only — no engine module imports the TEE-reaching set. (The former
    # attestation.py exemption is gone: that module was deleted when
    # attestation became a kernel-backed visibility view.)
    assert guards._gate_c_violations() == []


def test_full_trust_gate_passes_on_real_tree():
    # The whole trust gate as a regular local test (so pytest enforces it
    # before a PR, not only in CI). GREEN and baseline-free: any NEW
    # engine->enclave import fails this immediately.
    violations = (
        guards._gate_a_violations() + guards._gate_b_violations()
        + guards._gate_c_violations() + guards._gate_d_violations()
        + guards._gate_e_violations()
    )
    assert violations == [], "trust_gates found violations:\n" + "\n".join(
        violations
    )
