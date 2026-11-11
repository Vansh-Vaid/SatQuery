"""Evaluation metrics: VQA accuracy, captioning (BLEU-1..4, ROUGE-L, optional CIDEr), grounding IoU."""
from __future__ import annotations

import math
import re
from collections import Counter

_NUMBERS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
            "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def normalize_answer(a: str) -> str:
    a = str(a).lower().strip()
    a = re.sub(r"[^\w%\s\-_.]", " ", a)
    a = re.sub(r"\s+", " ", a).strip(" .")
    m = re.match(r"^(yes|no)\b", a)
    if m:
        return m.group(1)
    return " ".join(_NUMBERS.get(w, w) for w in a.split())


def quantize_count(a: str) -> str | None:
    """RSVQA-LR count protocol: 0 / 1-10 / 11-100 / 101-1000 / >1000. None if `a` is not a number."""
    m = re.search(r"-?\d+(\.\d+)?", str(a))
    if not m:
        return None
    n = float(m.group())
    return "0" if n <= 0 else "1-10" if n <= 10 else "11-100" if n <= 100 else "101-1000" if n <= 1000 else ">1000"


_CDVQA_TEXT = {"non-vegetated ground surface": "nvg_surface", "low vegetation": "low_vegetation"}


def canonical_cdvqa(a: str) -> str:
    """CDVQA labels vs SatQuery's readable answers: '0-10%' -> '0_to_10', '0%' -> '0'."""
    a = normalize_answer(a)
    a = _CDVQA_TEXT.get(a, a)
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)\s*%?", a)
    if m:
        return f"{m.group(1)}_to_{m.group(2)}"
    m = re.fullmatch(r"(\d+)\s*%", a)
    return m.group(1) if m else a


def vqa_correct(pred: str, gt: str, qtype: str | None = None) -> bool:
    if qtype in {"change_or_not", "increase_or_not", "decrease_or_not", "change_ratio", "change_ratio_types",
                 "change_to_what", "largest_change", "smallest_change"}:
        return canonical_cdvqa(pred) == canonical_cdvqa(gt)
    if qtype == "count":
        qp, qg = quantize_count(pred), quantize_count(gt)
        return qp is not None and qp == qg
    p, g = normalize_answer(pred), normalize_answer(gt)
    if p == g:
        return True
    mp = re.match(r"^\(?([a-e])\)?[\s.:)]", p + " ")   # MCQ letters: "B) forest" vs "b"
    mg = re.match(r"^\(?([a-e])\)?[\s.:)]", g + " ")
    if mp and mg:
        return mp.group(1) == mg.group(1)
    return bool(g) and len(g.split()) <= 3 and len(p.split()) <= 6 and re.search(rf"\b{re.escape(g)}\b", p) is not None


def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def bleu(pred: str, refs: list[str], n_max: int = 4) -> dict[str, float]:
    c = _tok(pred)
    rs = [_tok(r) for r in refs]
    out, logs = {}, []
    ref_len = min((len(r) for r in rs), key=lambda L: (abs(L - len(c)), L)) if rs else 0
    bp = 1.0 if len(c) > ref_len else math.exp(1 - ref_len / max(len(c), 1))
    for n in range(1, n_max + 1):
        cn = Counter(tuple(c[i:i + n]) for i in range(len(c) - n + 1))
        maxref: Counter = Counter()
        for r in rs:
            maxref |= Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
        clip = sum(min(v, maxref[k]) for k, v in cn.items())
        total = max(sum(cn.values()), 1)
        p = (clip + (1 if n > 1 else 0)) / (total + (1 if n > 1 else 0))  # add-one smoothing for n > 1
        logs.append(math.log(max(p, 1e-9)))
        out[f"BLEU-{n}"] = bp * math.exp(sum(logs) / n)
    return out


def rouge_l(pred: str, refs: list[str], beta: float = 1.2) -> float:
    c = _tok(pred)
    best = 0.0
    for r in (_tok(x) for x in refs):
        if not c or not r:
            continue
        dp = [[0] * (len(r) + 1) for _ in range(len(c) + 1)]
        for i in range(1, len(c) + 1):
            for j in range(1, len(r) + 1):
                dp[i][j] = dp[i - 1][j - 1] + 1 if c[i - 1] == r[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
        lcs = dp[-1][-1]
        if lcs:
            p, rc = lcs / len(c), lcs / len(r)
            best = max(best, (1 + beta ** 2) * p * rc / (rc + beta ** 2 * p))
    return best


def cider(preds: list[str], refs: list[list[str]]) -> float | None:
    """CIDEr via pycocoevalcap when installed (pip install pycocoevalcap); None otherwise."""
    try:
        from pycocoevalcap.cider.cider import Cider
    except ImportError:
        return None
    gts = {i: [" ".join(_tok(r)) for r in rs] for i, rs in enumerate(refs)}
    res = {i: [" ".join(_tok(p))] for i, p in enumerate(preds)}
    score, _ = Cider().compute_score(gts, res)
    return float(score)


def box_iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _default_label(c: float) -> str:
    # Same thresholds as satquery.agent.controller.confidence_label (kept here so eval needs no torch import).
    return "high" if c >= 0.75 else "medium" if c >= 0.5 else "low"


def calibration(confs: list[float], correct: list[bool], n_bins: int = 10, label_fn=_default_label) -> dict:
    """How well stated confidence matches accuracy.

    ECE = sum over equal-width bins of (fraction of answers in bin) * |mean confidence - accuracy|.
    Also returns the reliability-diagram bins and the accuracy behind each confidence label.
    """
    pairs = [(float(c), bool(y)) for c, y in zip(confs, correct) if c is not None]
    if not pairs:
        return {}
    n = len(pairs)
    bins = [[] for _ in range(n_bins)]
    for c, y in pairs:
        bins[min(int(c * n_bins), n_bins - 1)].append((c, y))
    rel, ece = [], 0.0
    for k, b in enumerate(bins):
        if not b:
            continue
        conf = sum(c for c, _ in b) / len(b)
        acc = sum(y for _, y in b) / len(b)
        ece += len(b) / n * abs(conf - acc)
        rel.append({"bin": [round(k / n_bins, 2), round((k + 1) / n_bins, 2)], "n": len(b),
                    "confidence": round(conf, 4), "accuracy": round(acc, 4)})
    labels: dict[str, list[int]] = {}
    for c, y in pairs:
        s = labels.setdefault(label_fn(c), [0, 0])
        s[0] += 1
        s[1] += int(y)
    return {"n": n, "ECE": round(ece, 4),
            "mean_confidence": round(sum(c for c, _ in pairs) / n, 4),
            "accuracy": round(sum(y for _, y in pairs) / n, 4),
            "by_label": {k: {"n": t, "accuracy": round(c / t, 4)} for k, (t, c) in sorted(labels.items())},
            "reliability": rel}
