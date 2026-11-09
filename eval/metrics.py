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


def vqa_correct(pred: str, gt: str, qtype: str | None = None) -> bool:
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
