#!/usr/bin/env python3
"""Trust-surface / supply-chain gates: source-level invariants that keep the
production binary from routing user data to a non-attested provider, leaking a
secret, or pulling test scaffolding into shipped code.

Run in the lint flow (lint.yml) AND as an early step in the release flow
(release.yml), so a release can never be cut from a ref the gate never
inspected. Each check fails the build (nonzero exit) printing the offending
``file:line``.

  (a) The substring ``fireworks`` may not appear anywhere in the repo root
      outside ``testing/``. The production binary routes user data only
      through TEE or LOCAL; even a stray mention is a smell, since the gate
      is the only structural defence against a future bug that wires the
      deleted provider path back in. ``testing/`` keeps its references
      (dev-test infra, never shipped).

  (b) Production code (``engine/``, ``kernel/``, ``src/``, ``src-tauri/``,
      ``scripts/``) may never import from ``testing/``. The dependency arrow
      goes one way: testing -> production. Without this a test helper could
      be pulled into a pipeline file, re-introducing a forbidden reference
      transitively.

  (c) The pipeline (``python/engine``) may never import the TEE-reaching set
      — cloud/TEE SDKs (T1), raw HTTP (T2), or attestation crypto (T3). All
      enclave access goes through the kernel's attested provider, so an
      un-attested call is structurally impossible. A documented baseline
      exempts the known-pending ``attestation.py`` sites (relocating into the
      kernel, #909 §5a/b) while still hard-failing any NEW site.

  (d) The substrings ``__import__`` and ``importlib`` may not appear in
      ``engine`` or ``kernel`` (case-insensitive). Dynamic imports name a
      module by an arbitrary expression, so a transport / ``testing`` import
      could hide from gates (b)/(c); banning the tokens keeps the import
      surface statically analyzable.

  (e) No dotenv may be tracked in git. ``.env`` files hold per-user / eval
      keys and are gitignored; once the repo is public a single ``git add
      -f`` would publish a live credential. Fails if git tracks any ``.env``.

Run with no arguments to scan the repo from its root. Exit 0 = clean.

Escape hatch (precise, not a blanket ban): an inline ``# ci-allow:net -
<reason>`` exempts one sanctioned stdlib-net use under gate (c) that does not
reach the enclave (e.g. the localhost Ollama daemon probe). Gates (a), (b),
(d), (e) have no escape hatch.

Correctness tripwires that are NOT trust-surface (e.g. the React ``listen()``
StrictMode guard) live in ``code_gates.py``.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Gate (a): no fireworks refs outside testing/ ---------------

APP_DIR = REPO_ROOT
TESTING_DIR = APP_DIR / "python" / "testing"
TRUST_GUT_TOKENS = re.compile(r"fireworks", re.IGNORECASE)

# This guard lives under scripts/ and must name the forbidden
# providers to forbid them — it's the one file outside testing/
# that legitimately contains the tokens. Exempt it from its own scan.
SELF_PATH = Path(__file__).resolve()

# File extensions worth scanning. Excluded: binary assets (.png, .icns, etc.)
# where a substring match would be noise from PNG metadata or similar.
GATE_A_EXTS = {
    ".py", ".rs", ".js", ".jsx", ".ts", ".tsx",
    ".sh", ".toml", ".json", ".md", ".yml", ".yaml", ".css", ".html",
}

# Vendored / build / venv trees walked from the repo root at dev time.
# None of them ship in the production binary; their contents (third-party
# wheels, Rust build artifacts, JS deps) are not the trust surface this
# gate defends. Skipping them keeps incidental string matches in
# upstream packages (e.g. `rich`'s emoji table contains "fireworks")
# from failing the build. `.git`/`.github` are skipped too: they are not
# the app trust surface, and the release workflow legitimately names the
# providers in comments.
GATE_A_SKIP_DIRS = {
    "target",       # Rust build (src-tauri/target/)
    "node_modules", # JS deps (node_modules/)
    ".venv",        # Python venv
    "binaries",     # Bundled python sidecar (src-tauri/binaries/)
    "__pycache__",
    ".git",         # version-control internals
    ".github",      # CI config (workflows name providers in comments)
}


def _under_skipped_dir(path: Path) -> bool:
    return any(part in GATE_A_SKIP_DIRS for part in path.parts)


def _gate_a_violations() -> list[str]:
    out: list[str] = []
    if not APP_DIR.exists():
        return out
    for path in sorted(APP_DIR.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in GATE_A_EXTS:
            continue
        if path.resolve() == SELF_PATH:
            continue
        if _under_skipped_dir(path.relative_to(REPO_ROOT)):
            continue
        try:
            path.resolve().relative_to(TESTING_DIR.resolve())
            # Inside testing/ — skip (testing is the sanctioned home).
            continue
        except ValueError:
            pass
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if TRUST_GUT_TOKENS.search(line):
                out.append(
                    f"{path.relative_to(REPO_ROOT)}:{i}: "
                    f"'fireworks' substring outside testing/ "
                    f"-- the production binary routes only through TEE or "
                    f"LOCAL; keep references inside testing/."
                )
    return out


# --- Gate (b): no testing.* imports / cross-tree references from prod tree -

PRODUCTION_DIRS = (
    APP_DIR / "python" / "engine",
    APP_DIR / "python" / "kernel",
    APP_DIR / "src",
    APP_DIR / "src-tauri",
    APP_DIR / "scripts",
)
PRODUCTION_EXTS = {".py", ".rs", ".js", ".jsx", ".ts", ".tsx"}

# Patterns recognized as production-tree code reaching into testing.
# Each language gets its own matcher because the syntax differs and a
# single regex would either miss real cases or fire on prose / comments.
GATE_B_PATTERNS: dict[str, re.Pattern] = {
    # Python: `from testing.<X> import <names>` or `import testing.<X>` /
    # `import testing` (with or without `as`). The from-branch requires a
    # space after `import` (the imported names follow, unanchored) — an
    # end-of-line anchor there would wrongly require `from testing import`
    # with NOTHING after it and so miss every real `from testing.x import
    # y`. The import-branch stays end-anchored so `import testing_utils`
    # (a different package) can't trip it.
    ".py": re.compile(
        r"^\s*(?:from\s+testing(?:\.\S+)?\s+import\s|"
        r"import\s+testing(?:\.\S+)?(?:\s+as\s+\w+)?\s*(?:$|#))"
    ),
    # Rust: `use testing::...` or `use crate::testing::...`. The crate
    # form is what a production module would write when reaching across
    # to a sibling-module sub-tree.
    ".rs": re.compile(r"^\s*use\s+(?:crate::)?testing(?:::\w+)+\s*;"),
    # JS / TS: `import ... from "../testing/..."` or relative paths
    # ending in `/testing/...`. We match the segment defensively so a
    # variable named `testing` in code doesn't trigger.
    ".js":  re.compile(r"""(?:from|require\()\s*["'][^"']*?/testing/"""),
    ".jsx": re.compile(r"""(?:from|require\()\s*["'][^"']*?/testing/"""),
    ".ts":  re.compile(r"""(?:from|require\()\s*["'][^"']*?/testing/"""),
    ".tsx": re.compile(r"""(?:from|require\()\s*["'][^"']*?/testing/"""),
}


def _gate_b_violations() -> list[str]:
    out: list[str] = []
    for prod_dir in PRODUCTION_DIRS:
        if not prod_dir.exists():
            continue
        for path in sorted(prod_dir.rglob("*")):
            if not path.is_file():
                continue
            if _under_skipped_dir(path.relative_to(REPO_ROOT)):
                continue
            pattern = GATE_B_PATTERNS.get(path.suffix)
            if pattern is None:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    out.append(
                        f"{path.relative_to(REPO_ROOT)}:{i}: "
                        f"production-tree file references `testing` -- "
                        f"the dependency arrow only goes one way "
                        f"(testing -> production); test infrastructure "
                        f"belongs under testing/."
                    )
    return out


# --- Gate (c): the pipeline (engine) never imports the TEE-reaching set -----
#
# The pipeline layer (python/engine — everything above job execution) may
# not import anything that reaches a model endpoint, does raw network I/O,
# or runs attestation/trust crypto. All of that goes through the kernel's
# attested provider, so an un-attested call is structurally impossible:
# no code above job execution can even import a transport that reaches the
# enclave. ENFORCING — a violation fails the build (#909).
#
# Scope is the engine, NOT scripts: scripts/ is dev/release tooling, not
# the user-data pipeline. The kernel is the sanctioned home for the
# transport set and is not scanned.
#
# The scan is AST-based (walks every Import / ImportFrom node, including
# imports nested inside functions) — a top-level `^import` grep would miss
# the lazy `from tinfoil import TinfoilAI` that llm.py / attestation.py do
# inside functions. The forbidden set is an explicit list on top of an
# otherwise-stdlib network surface; the precise inline carve-out
# `# ci-allow:net - <reason>` exempts a sanctioned stdlib-net use that does
# NOT reach the enclave (e.g. the localhost Ollama daemon health probe in
# setup_local.py). Local inference (ollama / mlx) is ALLOWED in the engine
# — local is in-process / localhost, nothing to attest.

GATE_C_PIPELINE_DIRS = (
    REPO_ROOT / "python" / "engine",
)
# Sub-dirs skipped inside the pipeline: test trees (test code may stub a
# provider) and bytecode caches.
GATE_C_SKIP_DIRS = {"tests", "__pycache__"}

# Forbidden module dotted-prefix -> (tier, reason). A prefix matches an
# import of itself or any submodule (`tinfoil` matches `tinfoil.x`;
# `urllib.request` matches it exactly but NOT the stdlib-safe
# `urllib.parse`). Tiers mirror the #909 deny-list. Local inference
# (Tier 3: ollama / mlx) is intentionally NOT here — local is allowed in
# the engine.
GATE_C_FORBIDDEN: dict[str, tuple[str, str]] = {
    # T1 — cloud / TEE model SDKs (the un-attested-call vector). Importing
    # the RAW SDK is forbidden; going through the kernel (e.g.
    # `from kernel.tinfoil_provider import TinfoilProvider`) is ALLOWED —
    # the kernel is the sanctioned, attested transport boundary.
    "tinfoil":          ("T1", "TEE transport SDK"),
    "openai":           ("T1", "cloud model SDK (raw base_url)"),
    "anthropic":        ("T1", "cloud model SDK"),
    "fireworks":        ("T1", "cloud model SDK"),
    "fireworks_ai":     ("T1", "cloud model SDK"),
    # T2 — raw HTTP (the tier that makes the gate real).
    "httpx":            ("T2", "raw HTTP"),
    "httpcore":         ("T2", "raw HTTP"),
    "requests":         ("T2", "raw HTTP"),
    "aiohttp":          ("T2", "raw HTTP"),
    "urllib3":          ("T2", "raw HTTP"),
    "http.client":      ("T2", "raw HTTP (stdlib)"),
    "urllib.request":   ("T2", "raw HTTP (stdlib)"),
    "urllib.error":     ("T2", "raw HTTP (stdlib)"),
    # T3 — attestation / trust crypto (kernel-owned).
    "sigstore":         ("T3", "attestation trust crypto"),
    "rfc3161_client":   ("T3", "attestation trust crypto"),
    "tuf":              ("T3", "attestation trust crypto"),
    "securesystemslib": ("T3", "attestation trust crypto"),
    "cryptography":     ("T3", "crypto (judgment)"),
    "OpenSSL":          ("T3", "pyopenssl (judgment)"),
}


def _gate_c_imported_paths(node: ast.AST) -> list[str]:
    """Dotted module path(s) a single import node brings into scope.

    For ``from X import a, b`` both ``X`` and ``X.a`` / ``X.b`` are
    returned, so a forbidden submodule is caught whether it's the
    `from`-target or one of the names. Relative imports (``level > 0``)
    return nothing — they cannot reach a top-level transport package."""
    paths: list[str] = []
    if isinstance(node, ast.Import):
        for a in node.names:
            paths.append(a.name)
    elif isinstance(node, ast.ImportFrom):
        if node.level:
            return paths
        mod = node.module or ""
        if mod:
            paths.append(mod)
            for a in node.names:
                paths.append(f"{mod}.{a.name}")
    return paths


def _gate_c_match(dotted: str) -> tuple[str, str, str] | None:
    """Return ``(prefix, tier, reason)`` if ``dotted`` hits a forbidden
    prefix, else None."""
    for prefix, (tier, why) in GATE_C_FORBIDDEN.items():
        if dotted == prefix or dotted.startswith(prefix + "."):
            return prefix, tier, why
    return None


def _gate_c_violations() -> list[str]:
    # The engine reaches the enclave through NOTHING but the kernel's attested
    # provider — no baseline exemptions. (The former GATE_C_BASELINE carved out
    # the engine-side attestation.py; that module was deleted once attestation
    # became a kernel-backed visibility view, so the gate is truly clean.)
    out: list[str] = []
    for base in GATE_C_PIPELINE_DIRS:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            rel = path.relative_to(base)
            if set(rel.parts[:-1]) & GATE_C_SKIP_DIRS:
                continue
            try:
                src = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            try:
                tree = ast.parse(src, filename=str(path))
            except SyntaxError:
                continue
            srclines = src.splitlines()
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                line_txt = (
                    srclines[node.lineno - 1]
                    if 0 < node.lineno <= len(srclines) else ""
                )
                # A precise inline carve-out exempts a sanctioned stdlib-net
                # use that does not reach the enclave (e.g. the localhost
                # Ollama daemon probe). It is the ONLY way past this gate.
                if "ci-allow:net" in line_txt:
                    continue
                seen: set[str] = set()
                for dotted in _gate_c_imported_paths(node):
                    hit = _gate_c_match(dotted)
                    if not hit or hit[0] in seen:
                        continue
                    prefix, tier, why = hit
                    seen.add(prefix)
                    rel_repo = str(path.relative_to(REPO_ROOT))
                    out.append(
                        f"{rel_repo}:{node.lineno}: "
                        f"[{tier}] pipeline imports `{dotted}` ({why}) -- "
                        f"the engine may not reach the enclave; route this "
                        f"through the kernel's attested provider."
                    )
    return out


# --- Gate (d): no dynamic-import tokens in engine or kernel -----------------
#
# The substrings ``__import__`` and ``importlib`` may not appear ANYWHERE in
# the engine or kernel trees (case-insensitive) — a hard token ban, exactly
# like gate (a)'s ``fireworks``, with no escape hatch. Both are dynamic-import
# mechanisms: ``__import__(name)`` / ``importlib.import_module(name)`` import a
# module named by an arbitrary expression, so a transport (or ``testing``)
# import can hide behind a variable / loop / env-var string that the static
# import gates (g, f) cannot resolve (e.g.
# ``for d in ("openai", "anthropic"): __import__(d)`` or
# ``importlib.import_module(os.environ["X"])``). Banning the tokens outright
# keeps the import surface of engine + kernel fully statically analyzable,
# which is the only thing that makes gates (g) and (f) sound. Tests under
# these trees are scanned too — there is no reason to spell an import that way.

GATE_D_DIRS = (
    REPO_ROOT / "python" / "engine",
    REPO_ROOT / "python" / "kernel",
)
GATE_D_SKIP_DIRS = {"__pycache__"}
GATE_D_TOKENS = re.compile(r"__import__|importlib", re.IGNORECASE)


def _gate_d_violations() -> list[str]:
    out: list[str] = []
    for base in GATE_D_DIRS:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            rel = path.relative_to(base)
            if set(rel.parts[:-1]) & GATE_D_SKIP_DIRS:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                m = GATE_D_TOKENS.search(line)
                if m:
                    out.append(
                        f"{path.relative_to(REPO_ROOT)}:{i}: "
                        f"'{m.group(0)}' is banned in engine/kernel -- dynamic "
                        f"imports hide the import surface from the trust gates; "
                        f"use a normal `import`."
                    )
    return out


# --- Gate (e): no tracked .env ----------------------------------------------

# Any tracked file whose basename ends in `.env` (covers `.env`, `oss/.env`,
# `foo.env`). `.env.example`-style templates end in `.example`, not `.env`.
GATE_E_DOTENV = re.compile(r"(?:^|/)[^/]*\.env$")


def _gate_e_violations() -> list[str]:
    try:
        tracked = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        # Not a git checkout (e.g. an unpacked release tarball) -> nothing
        # to assert about tracked files.
        return []
    return [
        f"{rel}: a dotenv is tracked -- `.env` files hold keys and must "
        f"stay gitignored. Remove with `git rm --cached`."
        for rel in tracked
        if GATE_E_DOTENV.search(rel)
    ]


def main() -> int:
    violations = (
        _gate_a_violations() + _gate_b_violations() + _gate_c_violations()
        + _gate_d_violations() + _gate_e_violations()
    )
    if not violations:
        print("trust_gates: clean (gates a + b + c + d + e)")
        return 0
    print("trust_gates: FAIL\n")
    for v in violations:
        print(f"  {v}")
    print(
        "\nFix the site, or for gate (c) add an inline `# ci-allow:net - "
        "<reason>` for a sanctioned stdlib-net use that does not reach the "
        "enclave. Gates a, b, d, e have no escape hatch: this trust-surface "
        "boundary is what keeps the production binary from routing user data "
        "to a non-attested provider or shipping a secret."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
