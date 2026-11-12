from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import check_numbers  # noqa: E402


def test_readme_numbers_match_the_results_files(capsys):
    rc = check_numbers.main(["--check"])
    assert rc == 0, capsys.readouterr().out
