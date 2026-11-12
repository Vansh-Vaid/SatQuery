"""The VLM fine-tuning schedule must finish (decay) inside the time limit."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from finetune_vlm import fit_total, lr_factor  # noqa: E402


def test_cosine_decays_to_zero_at_the_planned_end():
    assert lr_factor(0, 10, 100) == pytest.approx(0.1)
    assert lr_factor(10, 10, 100) == pytest.approx(1.0)
    assert lr_factor(100, 10, 100) == pytest.approx(0.0, abs=1e-12)
    # shrinking the plan mid-run moves the end of the decay with it
    assert lr_factor(50, 10, 50) == pytest.approx(0.0, abs=1e-12)


def test_plan_is_fitted_to_the_time_limit():
    # the last Kaggle run: 1.57 min/step, 389 min budget, 750 planned steps
    assert fit_total(750, 10, 15.7, 389) == 247
    assert fit_total(100, 10, 1.0, 389) == 100     # fast enough: never grows
    assert fit_total(750, 10, 15.7, 0) == 750      # no time limit
    assert fit_total(750, 240, 389.0, 389) == 241  # never ends before the next step
