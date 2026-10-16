"""Static docs must mark pre-version-3 counts as historical.

The clean package has no generated data, so current counts belong in the
runtime-generated reports. This check prevents old snapshots from looking
like current results; docs_sync separately checks numeric definition values.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from tactical_ilp import config

DOCS = Path(config.DOCS_DIR)


def _read(name: str) -> str:
    p = DOCS / name
    if not p.exists():
        pytest.skip(f"{name} missing")
    return p.read_text(encoding="utf-8")


def test_outcome_doc_does_not_present_old_counts_as_current():
    text = _read("negative_outcome_definition.md")
    assert "نسخهٔ قبلی" in text
    assert "باید در اجرای پاک Colab دوباره تولید شوند" in text


def test_sampling_and_setup_tables_are_marked_as_historical():
    sampling = _read("negative_sampling_strategy.md")
    text = _read("SETUP_WINDOWS.md")
    assert "برای نسخهٔ جاری نیستند" in sampling
    assert "اعداد snapshot نسخهٔ پیشین" in text
