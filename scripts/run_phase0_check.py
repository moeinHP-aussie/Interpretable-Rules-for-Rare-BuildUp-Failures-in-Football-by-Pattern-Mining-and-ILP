"""
Phase 0 — environment check.

Run this first, on your own machine, before anything else:

    uv run python scripts/run_phase0_check.py

It prints a checklist instead of raising, so you can see *everything* that is
missing in one pass rather than fixing one error at a time.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

OK = "[ OK ]"
NO = "[FAIL]"
WARN = "[WARN]"


def check(label: str, fn):
    try:
        ok, detail = fn()
    except Exception as exc:  # noqa: BLE001
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    print(f"{OK if ok else NO} {label:<42} {detail}")
    return ok


def _python():
    v = sys.version_info
    return (v >= (3, 11)), f"{platform.python_version()} ({platform.system()})"


def _packages():
    # This check runs in a headless `uv run` process, which may inherit
    # Colab's MPLBACKEND=module://matplotlib_inline.backend_inline.  Matplotlib
    # validates that setting while importing, before matplotlib.use() can
    # override it; remove it first, then select the non-interactive backend.
    os.environ.pop("MPLBACKEND", None)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import mplsoccer  # noqa: F401
    import numpy
    import pandas
    import socceraction
    import statsbombpy  # noqa: F401

    return True, f"pandas {pandas.__version__}, numpy {numpy.__version__}, socceraction {socceraction.__version__}"


def _pandera_stack():
    import socceraction.spadl as spadl

    assert hasattr(spadl.statsbomb, "convert_to_actions")
    from importlib.metadata import version

    v = version("multimethod")
    major = int(v.split(".")[0])
    return major < 2, f"multimethod {v} (must be < 2.0)"


def _swipl():
    exe = shutil.which("swipl")
    if exe is None:
        return False, "not on PATH"
    out = subprocess.run([exe, "--version"], capture_output=True, text=True)
    version = out.stdout.split("version")[1].split()[0]
    parts = tuple(int(p) for p in version.split(".")[:3])
    return parts >= (9, 1, 12), f"{version} (Popper needs >= 9.1.12)"


def _uv():
    exe = shutil.which("uv")
    if exe is None:
        return False, "not on PATH"
    out = subprocess.run([exe, "--version"], capture_output=True, text=True)
    return True, out.stdout.strip()


def _popper():
    env = os.environ.get("POPPER_DIR")
    root = Path(__file__).resolve().parents[1]
    for c in ([Path(env)] if env else []) + [root / "external" / "Popper", root.parent / "Popper"]:
        if c.exists() and (c / "popper.py").exists():
            return True, str(c)
    return False, "not found (set POPPER_DIR or clone into external/Popper)"


def _network():
    import urllib.request

    url = (
        "https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json"
    )
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status == 200, f"StatsBomb Open Data reachable (HTTP {r.status})"


def main() -> int:
    print("=" * 78)
    print("PHASE 0 — environment check")
    print("=" * 78)
    results = [
        check("Python >= 3.11", _python),
        check("core analysis packages", _packages),
        check("socceraction / pandera stack healthy", _pandera_stack),
        check("uv package manager", _uv),
        check("SWI-Prolog >= 9.1.12", _swipl),
        check("Popper checkout", _popper),
        check("network access to StatsBomb", _network),
    ]
    print("=" * 78)
    failed = results.count(False)
    if failed:
        print(f"{failed} check(s) failed — see history_docs/COLAB_GUIDE.md or history_docs/SETUP_WINDOWS.md")
    else:
        print("all checks passed — you can run scripts/run_phase1.py")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
