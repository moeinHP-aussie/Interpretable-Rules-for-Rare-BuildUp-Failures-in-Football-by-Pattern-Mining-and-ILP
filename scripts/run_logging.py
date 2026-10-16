"""Live console logging helpers for the Colab phase runners."""

from __future__ import annotations

import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


class _Tee:
    def __init__(self, console, log, lock: threading.Lock) -> None:
        self.console = console
        self.log = log
        self.lock = lock
        self.pending = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self.console.write(text)
        self.console.flush()
        with self.lock:
            self.pending += text
            while "\n" in self.pending:
                line, self.pending = self.pending.split("\n", 1)
                stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
                self.log.write(f"{stamp} {line}\n")
            self.log.flush()
        return len(text)

    def flush(self) -> None:
        self.console.flush()
        with self.lock:
            if self.pending:
                stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
                self.log.write(f"{stamp} {self.pending}\n")
                self.pending = ""
            self.log.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.console, "isatty", lambda: False)())

    @property
    def encoding(self):
        return getattr(self.console, "encoding", "utf-8")


def run_logged(phase: int | str, main: Callable[[], int]) -> int:
    """Run a phase with live notebook output mirrored to a timestamped log."""
    from tactical_ilp.config import REPORTS_DIR

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = str(phase).replace(" ", "_")
    log_path = REPORTS_DIR / f"{label}_run_{stamp}.log"
    started = time.monotonic()
    original_stdout, original_stderr = sys.stdout, sys.stderr
    lock = threading.Lock()
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        sys.stdout = _Tee(original_stdout, log, lock)
        sys.stderr = _Tee(original_stderr, log, lock)
        print(f"[run-log] live output is also saved to: {log_path}", flush=True)
        status = "failed"
        try:
            result = int(main())
            status = "completed" if result == 0 else f"failed (exit={result})"
            return result
        except BaseException as exc:
            print(f"[run-log] exception: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            raise
        finally:
            elapsed = time.monotonic() - started
            print(f"[run-log] {phase} {status}; elapsed={elapsed:.1f}s; log={log_path}", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = original_stdout, original_stderr
