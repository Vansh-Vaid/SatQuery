"""Calibration metric used to check the agent's confidence scores against benchmark accuracy."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from metrics import _default_label, calibration  # noqa: E402

from satquery.agent.controller import confidence_label  # noqa: E402


def test_perfectly_calibrated_scores_have_zero_ece():
    # 10 answers at 0.8 with 8 right, 10 at 0.3 with 3 right
    confs = [0.8] * 10 + [0.3] * 10
    correct = [True] * 8 + [False] * 2 + [True] * 3 + [False] * 7
    m = calibration(confs, correct)
    assert m["ECE"] == pytest.approx(0.0, abs=1e-9)
    assert m["accuracy"] == pytest.approx(0.55)
    assert m["by_label"]["high"] == {"n": 10, "accuracy": 0.8}
    assert m["by_label"]["low"] == {"n": 10, "accuracy": 0.3}


def test_overconfident_scores_have_positive_ece():
    m = calibration([0.95] * 4, [True, False, False, False])
    assert m["ECE"] == pytest.approx(0.70)
    assert m["reliability"] == [{"bin": [0.9, 1.0], "n": 4, "confidence": 0.95, "accuracy": 0.25}]


def test_missing_confidence_is_skipped_and_empty_input_is_empty():
    assert calibration([None, 1.0], [True, True])["n"] == 1
    assert calibration([], []) == {}


@pytest.mark.parametrize("c", [0.0, 0.49, 0.5, 0.74, 0.75, 0.99])
def test_eval_labels_match_agent_labels(c):
    assert _default_label(c) == confidence_label(c)


def test_count_questions_with_word_answers_are_compared_as_words():
    from metrics import vqa_correct
    assert vqa_correct("no", "no", "count")
    assert not vqa_correct("yes", "no", "count")
    assert vqa_correct("c", "c", "count") and not vqa_correct("a", "c", "count")
    assert vqa_correct("3", "3", "count") and vqa_correct("there are 7", "5", "count")  # RSVQA bins
