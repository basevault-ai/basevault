#!/usr/bin/env python3
"""Relocate Homebrew-linked libraries into the bundled Python sidecar.

Walks every .so / .dylib under <sidecar-dir>, finds dependencies that resolve
to a Homebrew prefix (/opt/homebrew/* or /usr/local/{opt,Cellar}/*), copies
them into <sidecar-dir>/lib/_vendor/, and rewrites the load paths in the
dependent files to use @rpath/<basename>. Adds an @loader_path-rooted rpath
so the dynamic loader finds the vendored copies.

After this runs the sidecar is fully self-contained: no runtime Homebrew
dependency, and no Team-ID hardened-runtime rejection on macOS notarized
.app bundles.

Usage: relocate-bundled-libs.py <sidecar-dir>

Idempotent: running twice does no further work (but must run BEFORE
recursive code-signing — install_name_tool invalidates signatures).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HOMEBREW_PREFIXES = (
    "/opt/homebrew/",
    "/usr/local/opt/",
    "/usr/local/Cellar/",
)

# Match the dep path on each line of `otool -L` output.
# Format: "\t<path> (compatibility version ..., current version ...)"
OTOOL_DEP_RE = re.compile(r"^\s+(\S+)\s+\(")


def is_homebrew(path: str) -> bool:
    return any(path.startswith(p) for p in HOMEBREW_PREFIXES)


def list_load_deps(file: Path) -> list[str]:
    """Return all dynamic-load deps of `file` (regardless of prefix)."""
    try:
        out = subprocess.run(
            ["otool", "-L", str(file)],
            capture_output=True, text=True, check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return []
    deps: list[str] = []
    for line in out.splitlines()[1:]:  # first line echoes the file path
        m = OTOOL_DEP_RE.match(line)
        if m:
            deps.append(m.group(1))
    return deps


def list_homebrew_deps(file: Path) -> list[str]:
    return [d for d in list_load_deps(file) if is_homebrew(d)]


def has_rpath(file: Path, rpath: str) -> bool:
    try:
        out = subprocess.run(
            ["otool", "-l", str(file)],
            capture_output=True, text=True, check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return False
    # `otool -l` prints LC_RPATH commands with `path <rpath>` on a separate line.
    return f"path {rpath}" in out


def make_writable(file: Path) -> None:
    os.chmod(file, 0o755)


def install_name_tool(*args: str) -> None:
    subprocess.run(
        ["install_name_tool", *args],
        check=True, capture_output=True,
    )


def relocate(sidecar_dir: Path) -> int:
    vendor_dir = sidecar_dir / "lib" / "_vendor"
    vendor_dir.mkdir(parents=True, exist_ok=True)

    # Discover Mach-O files in the sidecar (.so + .dylib).
    macho_files: list[Path] = sorted(
        f for f in sidecar_dir.rglob("*")
        if f.is_file() and f.suffix in (".so", ".dylib")
    )
    print(f"Scanning {len(macho_files)} Mach-O files under {sidecar_dir}")

    # Pass 1: build the transitive closure of Homebrew deps to vendor.
    #         keyed by ORIGINAL path (the reference string used in load
    #         commands; symlinks resolve via Path.resolve when copying).
    vendored: dict[str, Path] = {}
    pending: set[str] = set()
    for f in macho_files:
        pending.update(list_homebrew_deps(f))

    while pending:
        dep = pending.pop()
        if dep in vendored:
            continue
        real = Path(dep).resolve()
        if not real.is_file():
            print(f"  ! skip (not found): {dep}", file=sys.stderr)
            continue
        target = vendor_dir / Path(dep).name
        if not target.exists():
            shutil.copy2(real, target)
            make_writable(target)
            print(f"  vendored: {dep}")
        vendored[dep] = target
        # transitive deps of the vendored copy
        for transitive in list_homebrew_deps(target):
            if transitive not in vendored:
                pending.add(transitive)

    if not vendored:
        print("No Homebrew dependencies found — sidecar is already clean.")
        return 0

    # Pass 2: rewrite every consumer file to use @rpath/<basename>.
    all_to_fix = list(macho_files) + list(vendored.values())
    for f in all_to_fix:
        homebrew_deps = list_homebrew_deps(f)
        if not homebrew_deps:
            continue
        make_writable(f)
        for dep in homebrew_deps:
            new = f"@rpath/{Path(dep).name}"
            install_name_tool("-change", dep, new, str(f))

    # Pass 3: each vendored library's own ID becomes @rpath/<basename>.
    for original, target in vendored.items():
        new_id = f"@rpath/{target.name}"
        make_writable(target)
        install_name_tool("-id", new_id, str(target))

    # Pass 4: add an rpath to each consumer so it can find the vendor dir.
    #         For .so files in site-packages: rpath = @loader_path/<rel>.
    #         For vendored libs themselves: rpath = @loader_path (same dir).
    for f in macho_files:
        if not list_load_deps(f):
            continue
        # All consumer files should be able to find vendor_dir.
        rel = os.path.relpath(vendor_dir, f.parent)
        rpath = f"@loader_path/{rel}"
        if has_rpath(f, rpath):
            continue
        make_writable(f)
        try:
            install_name_tool("-add_rpath", rpath, str(f))
        except subprocess.CalledProcessError:
            # add_rpath fails noisily if rpath already exists; silently OK.
            pass

    for target in vendored.values():
        rpath = "@loader_path"
        if has_rpath(target, rpath):
            continue
        make_writable(target)
        try:
            install_name_tool("-add_rpath", rpath, str(target))
        except subprocess.CalledProcessError:
            pass

    # Final verification: re-scan, ensure no Homebrew deps remain.
    remaining = []
    for f in all_to_fix:
        deps = list_homebrew_deps(f)
        if deps:
            remaining.append((f, deps))
    if remaining:
        print("\nERROR: Homebrew dependencies still present after relocation:",
              file=sys.stderr)
        for f, deps in remaining:
            print(f"  {f}", file=sys.stderr)
            for d in deps:
                print(f"    -> {d}", file=sys.stderr)
        return 1

    unique_targets = sorted({p.name for p in vendored.values()})
    print(f"\nOK: vendored {len(unique_targets)} libraries into "
          f"{vendor_dir.relative_to(sidecar_dir)}")
    print("    " + ", ".join(unique_targets))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <sidecar-dir>", file=sys.stderr)
        sys.exit(2)
    sidecar = Path(sys.argv[1]).resolve()
    if not sidecar.is_dir():
        print(f"not a directory: {sidecar}", file=sys.stderr)
        sys.exit(2)
    sys.exit(relocate(sidecar))
