"""Bi-temporal change specialists.

  change_detector   where did it change? (normalised CVA for optical, log-ratio for SAR,
                    or the learned Siamese U-Net when a checkpoint is configured)
  change_semantics  what changed into what? (land-cover transitions inside the change mask)
  change_qa         answers change questions / writes the change description from the evidence
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import numpy as np

from satquery.config import settings
from satquery.imagery.raster import Modality, Raster
from satquery.imagery.validate import InputConfig
from satquery.tools.base import Box, Layer, ParamSpec, Tool, ToolContext, ToolResult, ToolSpec, TaskType
from satquery.tools.landcover import classify, clean_mask, lee_filter, otsu, regions, spatial_distribution
from satquery.tools.vocab import CLASS_COLORS, CLASS_LABELS, CLASSES, find_classes

BITEMP = (InputConfig.BITEMPORAL,)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%" if x >= 0.001 or x == 0 else "<0.1%"


def _dates(ctx: ToolContext) -> tuple[str, str]:
    a, b = ctx.rasters
    return (a.acquired.isoformat() if a.acquired else "date 1", b.acquired.isoformat() if b.acquired else "date 2")


# =================================================================== detector
_OPTICAL_ORDER = ["B02", "B03", "B04", "B08", "B11", "B12"]


def _common_optical(a: Raster, b: Raster) -> list[str]:
    return [n for n in _OPTICAL_ORDER if a.has(n) and b.has(n)]


def _match(src: np.ndarray, ref: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Robust linear radiometric normalisation of src to ref (median / IQR matching)."""
    s, r = src[valid], ref[valid]
    if s.size < 10:
        return src
    sq = np.percentile(s, [25, 50, 75])
    rq = np.percentile(r, [25, 50, 75])
    scale = (rq[2] - rq[0]) / max(sq[2] - sq[0], 1e-6)
    scale = float(np.clip(scale, 0.5, 2.0))
    return (src - sq[1]) * scale + rq[1]


@lru_cache(maxsize=1)
def _learned_model(path: str):  # pragma: no cover - needs a checkpoint
    from satquery.models.change_net import load_change_net
    return load_change_net(path)


class ChangeDetector(Tool):
    spec = ToolSpec(
        name="change_detector",
        title="Bi-temporal change detector",
        description="Produces a binary change map, change-magnitude heatmap and changed regions for two images "
                    "of the same area. Optical: radiometrically normalised change-vector analysis; SAR: Lee-filtered "
                    "log-ratio; optional learned Siamese U-Net (LEVIR-CD) when a checkpoint is configured. Adaptive "
                    "Otsu threshold with a physical floor so unchanged pairs are not forced to show change.",
        tasks=(TaskType.CHANGE_VQA, TaskType.CHANGE_DESCRIPTION),
        input_configs=BITEMP,
        kind="algorithmic / learned (optional)",
        params=(
            ParamSpec("method", "enum", "auto", "auto | cva | log_ratio | learned",
                      choices=("auto", "cva", "log_ratio", "learned")),
            ParamSpec("sensitivity", "float", 1.0, "Threshold divisor (>1 detects subtler change)", 0.5, 2.0),
            ParamSpec("min_region_px", "int", 0, "Smallest change region (0 = 0.05% of image, min 6 px)", 0, 100000),
        ),
    )

    def run(self, ctx: ToolContext, params: dict[str, Any]) -> ToolResult:
        a, b = ctx.rasters
        valid = a.valid_mask & b.valid_mask
        sens = float(params.get("sensitivity", 1.0))
        method = params.get("method", "auto")
        sar = a.modality is Modality.SAR
        notes: list[str] = []
        if method == "learned" or (method == "auto" and settings.change_checkpoint and not sar):
            if settings.change_checkpoint:
                return self._learned(ctx, a, b, valid, params)
            notes.append("Learned change model not configured (SATQUERY_CHANGE_CKPT); used algorithmic method.")
            method = "auto"
        if sar or method == "log_ratio":
            if not sar:
                notes.append("log_ratio requested for optical data; using CVA instead.")
                mag, thr, desc = self._cva(a, b, valid, sens)
            else:
                mag, thr, desc = self._log_ratio(a, b, valid, sens)
        else:
            mag, thr, desc = self._cva(a, b, valid, sens)
        mask = (mag > thr) & valid
        return self._finish(ctx, mask, mag, thr, valid, desc, params, notes,
                            scale=thr * 0.35 if thr > 0 else 1.0)

    # ---------------------------------------------------------------- methods
    def _cva(self, a: Raster, b: Raster, valid: np.ndarray, sens: float) -> tuple[np.ndarray, float, str]:
        names = _common_optical(a, b) or [None]
        if names == [None]:
            n = min(a.data.shape[0], b.data.shape[0], 3)
            pa, pb = [a.data[i] for i in range(n)], [b.data[i] for i in range(n)]
            names = [f"band{i + 1}" for i in range(n)]
        else:
            pa, pb = [a.band(n) for n in names], [b.band(n) for n in names]
        z2 = np.zeros(valid.shape, np.float32)
        absmax = np.zeros(valid.shape, np.float32)
        for x, y in zip(pa, pb):
            x = np.nan_to_num(x)
            y = _match(np.nan_to_num(y), x, valid)
            d = y - x
            sd = float(np.std(x[valid])) or 1e-3
            z2 += (d / sd) ** 2
            absmax = np.maximum(absmax, np.abs(d))
        mag = np.sqrt(z2 / len(pa))
        # pixels whose reflectance barely moved cannot be "changed", whatever their z-score
        mag = np.where(absmax < 0.03, np.minimum(mag, 0.5), mag)
        hi = float(np.percentile(mag[valid], 99.5)) if valid.any() else 1.0
        t = otsu(mag[valid], 0.0, max(hi, 1e-3))
        floor = 1.2
        thr = max(t if t is not None else floor, floor) / sens
        return mag, thr, f"normalised change-vector analysis on {', '.join(names)} (Otsu threshold, floor {floor}σ)"

    def _log_ratio(self, a: Raster, b: Raster, valid: np.ndarray, sens: float) -> tuple[np.ndarray, float, str]:
        pols = [p for p in ("VV", "VH") if a.has(p) and b.has(p)] or ["VV"]
        acc = np.zeros(valid.shape, np.float32)
        for p in pols:
            fa = a.band(p) if a.has(p) else a.data[0]
            fb = b.band(p) if b.has(p) else b.data[0]
            la = 10 * np.log10(np.maximum(lee_filter(10 ** (np.nan_to_num(fa, nan=-25) / 10)), 1e-6))
            lb = 10 * np.log10(np.maximum(lee_filter(10 ** (np.nan_to_num(fb, nan=-25) / 10)), 1e-6))
            acc += np.abs(lb - la)
        mag = acc / len(pols)
        t = otsu(mag[valid], 0.0, 15.0)
        floor = 3.0
        thr = max(t if t is not None else floor, floor) / sens
        return mag, thr, f"Lee-filtered log-ratio on {'+'.join(pols)} (Otsu threshold, floor {floor} dB)"

    def _learned(self, ctx: ToolContext, a: Raster, b: Raster, valid: np.ndarray,
                 params: dict[str, Any]) -> ToolResult:  # pragma: no cover - needs checkpoint
        import torch
        from satquery.imagery.render import to_rgb
        model = _learned_model(settings.change_checkpoint)
        ta = torch.from_numpy(to_rgb(a)).permute(2, 0, 1)[None].float() / 255
        tb = torch.from_numpy(to_rgb(b)).permute(2, 0, 1)[None].float() / 255
        with torch.no_grad():
            prob = torch.sigmoid(model(ta, tb))[0, 0].numpy()
        thr = 0.5 / float(params.get("sensitivity", 1.0))
        return self._finish(ctx, (prob > thr) & valid, prob, thr, valid,
                            f"learned Siamese U-Net ({settings.change_checkpoint})", params, [], scale=0.15)

    def _finish(self, ctx: ToolContext, mask: np.ndarray, mag: np.ndarray, thr: float, valid: np.ndarray,
                desc: str, params: dict[str, Any], notes: list[str], scale: float) -> ToolResult:
        h, w = mask.shape
        min_px = int(params.get("min_region_px") or max(6, int(0.0005 * h * w)))
        mask = clean_mask(mask, 1)
        regs = regions(mask, min_px=min_px, label="change")
        keep = np.zeros_like(mask)
        if regs:
            import cv2
            _, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
            ids = np.array([r["id"] for r in regs])
            keep = np.isin(lab, ids)
        frac = float(keep.sum()) / max(int(valid.sum()), 1)
        # confidence: how decisively pixels sit on either side of the threshold
        sig = 1 / (1 + np.exp(-(mag - thr) / max(scale, 1e-6)))
        c_changed = float(np.mean(sig[keep])) if keep.any() else 1.0
        c_static = float(np.mean(1 - sig[valid & ~keep])) if (valid & ~keep).any() else 1.0
        conf = 0.5 * (c_changed + c_static) if keep.any() else c_static
        d1, d2 = _dates(ctx)
        boxes = [Box(r["x0"], r["y0"], r["x1"], r["y1"], label=f"change {i + 1}", source=self.spec.name,
                     score=float(np.mean(sig[r["y0"]:r["y1"], r["x0"]:r["x1"]])))
                 for i, r in enumerate(regs[:10])]
        facts = {"method": desc, "threshold": round(float(thr), 3), "changed_pct": round(100 * frac, 2),
                 "n_regions": len(regs), "distribution": spatial_distribution(keep),
                 "dates": [d1, d2], "min_region_px": min_px,
                 "largest_regions": [{k: (round(v, 4) if isinstance(v, float) else v)
                                      for k, v in r.items() if k not in ("id", "label")} for r in regs[:5]]}
        ans = (f"{_pct(frac)} of the area changed between {d1} and {d2} ({len(regs)} region(s), "
               f"{facts['distribution']}).") if regs else f"No significant change detected between {d1} and {d2}."
        layers = [Layer("change mask", "mask", keep, color=CLASS_COLORS["change"], legend="detected change"),
                  Layer("change magnitude", "heatmap", np.where(valid, mag, np.nan), cmap="magma", vmin=0,
                        vmax=float(max(thr * 2, 1e-3)), legend="change magnitude (threshold at mid-scale)")]
        return ToolResult(self.spec.name, answer=ans, facts=facts, layers=layers, boxes=boxes,
                          confidence=float(np.clip(conf, 0.05, 0.98)), notes=notes,
                          data={"mask": keep, "magnitude": mag, "regions": regs})


# =================================================================== semantics
class ChangeSemantics(Tool):
    spec = ToolSpec(
        name="change_semantics",
        title="Land-cover transition analyser",
        description="Classifies both dates into water/vegetation/built-up/bare soil and measures class "
                    "transitions inside the detected change mask: per-class gain/loss and the dominant "
                    "from->to transitions (e.g. vegetation -> built-up).",
        tasks=(TaskType.CHANGE_VQA, TaskType.CHANGE_DESCRIPTION),
        input_configs=BITEMP,
        kind="algorithmic",
        requires=("change_detector",),
        params=(ParamSpec("min_delta_pp", "float", 0.5, "Net change (percentage points) below which a class is "
                                                        "reported as unchanged", 0.0, 10.0),),
    )

    def run(self, ctx: ToolContext, params: dict[str, Any]) -> ToolResult:
        a, b = ctx.rasters
        det = ctx.upstream.get("change_detector")
        if det is None or not det.ok:
            det = ChangeDetector().run(ctx, {})
        cmask = det.data["mask"]
        la, lb = classify(a), classify(b)
        valid = la.valid & lb.valid
        n = max(int(valid.sum()), 1)
        mapa, mapb = la.label_map(), lb.label_map()
        ch = cmask & valid & (mapa >= 0) & (mapb >= 0)
        tol = float(params.get("min_delta_pp", 0.5))
        per_class: dict[str, Any] = {}
        for i, c in enumerate(CLASSES):
            gain = float((ch & (mapb == i) & (mapa != i)).sum()) / n
            loss = float((ch & (mapa == i) & (mapb != i)).sum()) / n
            before = float(((mapa == i) & valid).sum()) / n
            net = gain - loss
            after = before + net  # change-mask-consistent estimate (ignores classifier noise outside changes)
            rel = abs(net) / max(before, 1e-6)
            if abs(net) * 100 >= tol and (rel >= 0.05 or before < 1e-3):
                direction = "increased" if net > 0 else "decreased"
            else:
                direction = "unchanged"
            per_class[c] = {"before_pct": round(100 * before, 2), "after_pct": round(100 * after, 2),
                            "gain_pct": round(100 * gain, 2), "loss_pct": round(100 * loss, 2),
                            "net_pp": round(100 * net, 2), "relative_change_pct": round(100 * rel, 1),
                            "direction": direction}
        trans = []
        layers = []
        for i, c1 in enumerate(CLASSES):
            for j, c2 in enumerate(CLASSES):
                if i == j:
                    continue
                m = ch & (mapa == i) & (mapb == j)
                f = float(m.sum()) / n
                if f > 0:
                    trans.append({"from": c1, "to": c2, "pct": round(100 * f, 2),
                                  "where": spatial_distribution(m), "_mask": m})
        trans.sort(key=lambda t: t["pct"], reverse=True)
        for t in trans[:3]:
            if t["pct"] >= 0.1:
                layers.append(Layer(f"transition: {CLASS_LABELS[t['from']]} → {CLASS_LABELS[t['to']]}", "mask",
                                    t["_mask"], color=CLASS_COLORS[t["to"]],
                                    legend=f"{CLASS_LABELS[t['from']]} → {CLASS_LABELS[t['to']]}"))
        unexplained = float((cmask & valid & ((mapa == mapb) | (mapa < 0) | (mapb < 0))).sum()) / n
        facts = {"method": f"post-classification comparison inside change mask (date 1: {la.method}; date 2: {lb.method})",
                 "per_class": per_class,
                 "top_transitions": [{k: v for k, v in t.items() if k != "_mask"} for t in trans[:6]],
                 "change_without_class_transition_pct": round(100 * unexplained, 2)}
        if trans:
            t = trans[0]
            ans = (f"Main transition: {CLASS_LABELS[t['from']]} → {CLASS_LABELS[t['to']]} ({t['pct']:.1f}% of the "
                   f"scene, {t['where']}).")
        else:
            ans = "No land-cover class transitions inside the detected change areas."
        moved = [f"{CLASS_LABELS[c]} {v['direction']} ({v['net_pp']:+.1f} pp)" for c, v in per_class.items()
                 if v["direction"] != "unchanged"]
        if moved:
            ans += " " + "; ".join(moved) + "."
        conf = float(np.mean([la.class_confidence(c) for c in CLASSES] + [lb.class_confidence(c) for c in CLASSES]))
        conf *= det.confidence or 0.7
        return ToolResult(self.spec.name, answer=ans, facts=facts, layers=layers, confidence=float(conf ** 0.5),
                          data={"per_class": per_class, "transitions": trans})


# =================================================================== QA
_YESNO = re.compile(r"^\s*(is|are|was|were|has|have|had|did|does|do)\b")
_ANY_CHANGE = re.compile(r"\b(chang|differ|modif|alter|transform)")
_DIRECTION = re.compile(r"\b(increase|decrease|grow|grew|shrink|shrank|expand|expan|reduc|declin|gain|lost|loss|"
                        r"lose|remain|unchanged|same|more|less)\w*")
_RATIO = re.compile(r"\b(how much|what (percentage|percent|ratio|proportion|fraction)|ratio|proportion|extent|"
                    r"how large|how big|amount)\b")
_LARGEST = re.compile(r"\b(largest|biggest|most|major|main|dominant|greatest)\b")
_SMALLEST = re.compile(r"\b(smallest|least|minor)\b")
_INTO = re.compile(r"\b(into what|to what|turn(ed|s)? (in)?to|convert(ed|s)? (in)?to|become|became|replaced)\b")
_WHERE = re.compile(r"\b(where|location|located|which part)\b")
_DESCRIBE = re.compile(r"\b(what (has )?changed|describe|summar|what happened|what are the changes|explain)\b")


def ratio_bin(pct: float) -> str:
    """CDVQA-style change-ratio bin, e.g. 12.3 -> '10%_to_20%'."""
    if pct <= 0:
        return "0%"
    lo = min(int(pct // 10) * 10, 90)
    return f"{lo}%_to_{lo + 10}%"


class ChangeQA(Tool):
    spec = ToolSpec(
        name="change_qa",
        title="Change question answerer",
        description="Answers change questions (is there change / did class X increase, decrease or stay the "
                    "same / how much changed / largest change / what did X turn into / where) and writes the "
                    "change description, using change_detector and change_semantics evidence.",
        tasks=(TaskType.CHANGE_VQA, TaskType.CHANGE_DESCRIPTION),
        input_configs=BITEMP,
        kind="reasoning",
        requires=("change_detector", "change_semantics"),
        params=(ParamSpec("mode", "enum", "auto", "qa | describe | auto", choices=("auto", "qa", "describe")),
                ParamSpec("presence_threshold_pct", "float", 0.5, "Changed share above which 'is there change' "
                                                                  "is answered yes", 0.0, 20.0)),
    )

    def run(self, ctx: ToolContext, params: dict[str, Any]) -> ToolResult:
        det = ctx.upstream.get("change_detector")
        sem = ctx.upstream.get("change_semantics")
        if det is None or not det.ok:
            det = ChangeDetector().run(ctx, {})
            ctx.upstream["change_detector"] = det
        if sem is None or not sem.ok:
            sem = ChangeSemantics().run(ctx, {})
        q = ctx.query.lower().strip()
        df, sf = det.facts, sem.facts
        pc = sf["per_class"]
        trans = sf["top_transitions"]
        changed = df["changed_pct"]
        thr = float(params.get("presence_threshold_pct", 0.5))
        classes = [c for c in find_classes(q) if c in CLASSES]
        base_conf = float(np.sqrt((det.confidence or 0.6) * (sem.confidence or 0.6)))
        mode = params.get("mode", "auto")
        facts = {"changed_pct": changed, "change_ratio_bin": ratio_bin(changed)}

        def out(short: str, text: str, qtype: str, conf: float) -> ToolResult:
            return ToolResult(self.spec.name, answer=text, confidence=float(np.clip(conf, 0.05, 0.97)),
                              facts={**facts, "question_type": qtype, "short_answer": short})

        if mode != "describe" and _INTO.search(q) and classes:
            c = classes[0]
            cand = [t for t in trans if t["from"] == c]
            if not cand:
                return out("no change", f"{CLASS_LABELS[c].capitalize()} did not convert to another class.",
                           "transition", base_conf * 0.8)
            t = cand[0]
            return out(CLASS_LABELS[t["to"]], f"{CLASS_LABELS[c].capitalize()} mainly became {CLASS_LABELS[t['to']]} "
                                              f"({t['pct']:.1f}% of the scene, {t['where']}).", "transition", base_conf)

        if mode == "describe" or (mode == "auto" and (_DESCRIBE.search(q) or _WHERE.search(q)) and not _YESNO.match(q)):
            return out("description", self._describe(ctx, df, sf), "describe", base_conf)

        if (_LARGEST.search(q) or _SMALLEST.search(q)) and re.search(r"\b(change|changed|area|class|type)\b", q) \
                and not classes:
            ranked = sorted(pc.items(), key=lambda kv: abs(kv[1]["net_pp"]), reverse=not _SMALLEST.search(q))
            c, v = ranked[0]
            word = "smallest" if _SMALLEST.search(q) else "largest"
            return out(CLASS_LABELS[c], f"The {word} net change is in {CLASS_LABELS[c]} ({v['net_pp']:+.1f} pp: "
                                        f"{v['before_pct']:.1f}% → {v['after_pct']:.1f}%).", "extreme", base_conf * 0.9)

        if _DIRECTION.search(q) and classes:
            c = classes[0]
            v = pc[c]
            d = v["direction"]
            word = {"increased": "increased", "decreased": "decreased", "unchanged": "remained unchanged"}[d]
            short = word
            if _YESNO.match(q):  # "has the built-up area increased?" -> yes/no
                asked_up = bool(re.search(r"increas|grow|grew|expan|gain|more", q))
                asked_down = bool(re.search(r"decreas|shrink|shrank|reduc|declin|lost|loss|lose|less", q))
                asked_same = bool(re.search(r"remain|unchanged|same", q))
                if asked_up + asked_down + asked_same == 1:
                    hit = (asked_up and d == "increased") or (asked_down and d == "decreased") or \
                          (asked_same and d == "unchanged")
                    short = "yes" if hit else "no"
            text = (f"The {CLASS_LABELS[c]} {word}: {v['before_pct']:.1f}% → {v['after_pct']:.1f}% of the scene "
                    f"({v['net_pp']:+.1f} pp; gained {v['gain_pct']:.1f}%, lost {v['loss_pct']:.1f}%).")
            margin = min(1.0, abs(v["net_pp"]) / max(float(pc[c].get("before_pct", 1)) * 0.1, 0.5))
            conf = base_conf * (0.75 + 0.25 * margin) if d != "unchanged" else base_conf * 0.8
            return out(short, text, "direction", conf)

        if _RATIO.search(q):
            if classes:
                c = classes[0]
                v = pc[c]
                return out(f"{v['gain_pct'] + v['loss_pct']:.1f}%",
                           f"{CLASS_LABELS[c].capitalize()} changed on {v['gain_pct'] + v['loss_pct']:.1f}% of the scene "
                           f"(gain {v['gain_pct']:.1f}%, loss {v['loss_pct']:.1f}%).", "ratio", base_conf)
            return out(ratio_bin(changed), f"{changed:.1f}% of the area changed (bin {ratio_bin(changed)}).", "ratio",
                       base_conf)

        if _YESNO.match(q) and (_ANY_CHANGE.search(q) or classes):
            if classes:
                c = classes[0]
                v = pc[c]
                touched = v["gain_pct"] + v["loss_pct"]
                yes = touched >= thr
                text = (f"{'Yes' if yes else 'No'}: {CLASS_LABELS[c]} changed on {touched:.1f}% of the scene "
                        f"(net {v['net_pp']:+.1f} pp).")
                return out("yes" if yes else "no", text, "presence", base_conf)
            yes = changed >= thr
            return out("yes" if yes else "no", f"{'Yes' if yes else 'No'}: {changed:.1f}% of the area changed "
                                               f"({df['n_regions']} region(s)).", "presence", base_conf)

        if mode == "qa":
            return ToolResult(self.spec.name, "unsupported", facts=facts,
                              notes=["Change question type not recognised; returning a description instead."])
        return out("description", self._describe(ctx, df, sf), "describe", base_conf * 0.9)

    @staticmethod
    def _describe(ctx: ToolContext, df: dict, sf: dict) -> str:
        d1, d2 = df["dates"]
        if df["n_regions"] == 0:
            return f"No significant change was detected between {d1} and {d2}."
        parts = [f"About {df['changed_pct']:.1f}% of the area changed between {d1} and {d2}, in "
                 f"{df['n_regions']} region(s) {df['distribution']}."]
        big = df.get("largest_regions") or []
        if big:
            parts.append("The largest changed region lies in the " + big[0]["direction"] +
                         f" ({100 * big[0]['area_frac']:.1f}% of the scene).")
        tr = [t for t in sf["top_transitions"] if t["pct"] >= 0.1][:3]
        if tr:
            parts.append("Main transitions: " + "; ".join(
                f"{CLASS_LABELS[t['from']]} → {CLASS_LABELS[t['to']]} ({t['pct']:.1f}%, {t['where']})" for t in tr) + ".")
        moved = [f"{CLASS_LABELS[c]} {v['direction']} ({v['before_pct']:.1f}% → {v['after_pct']:.1f}%)"
                 for c, v in sf["per_class"].items() if v["direction"] != "unchanged"]
        if moved:
            parts.append("Net effect: " + "; ".join(moved) + ".")
        if sf.get("change_without_class_transition_pct", 0) > 0.5 * df["changed_pct"]:
            parts.append("Much of the change is within a class (e.g. seasonal or illumination differences, crop "
                         "cycles or new structures of the same cover type).")
        return " ".join(parts)
