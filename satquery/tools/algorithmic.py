"""Single-image and cross-modal algorithmic specialists.

  landcover_optical   spectral land-cover mapping (S2 / RGB)
  landcover_sar       SAR backscatter land-cover mapping (S1)
  optical_sar_fusion  confidence-weighted decision fusion of the two + cloud gap-filling
  region_grounder     text-guided region grounding over land-cover masks
  landcover_reasoner  question answering / scene description over land-cover evidence
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from satquery.imagery.raster import Modality, Raster
from satquery.imagery.validate import InputConfig
from satquery.tools.base import Box, Layer, ParamSpec, Tool, ToolContext, ToolResult, ToolSpec, TaskType
from satquery.tools.landcover import (LandCoverMap, classify_optical, classify_sar, clean_mask, regions,
                                      spatial_distribution)
from satquery.tools.vocab import (CLASS_COLORS, CLASS_LABELS, CLASSES, find_classes, find_qualifiers)

SINGLE_OR_PAIR = (InputConfig.SINGLE, InputConfig.CROSS_MODAL)


def _optical(ctx: ToolContext) -> Raster | None:
    return next((r for r in ctx.rasters if r.modality.is_optical), None)


def _sar(ctx: ToolContext) -> Raster | None:
    return next((r for r in ctx.rasters if r.modality is Modality.SAR), None)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%" if x >= 0.001 or x == 0 else "<0.1%"


def _area_text(frac: float, r: Raster) -> str:
    pa = r.pixel_area_m2
    if pa is None:
        return _pct(frac)
    h, w = r.shape
    km2 = frac * h * w * pa / 1e6
    return f"{_pct(frac)} (~{km2:.2f} km²)" if km2 >= 0.01 else f"{_pct(frac)} (~{km2 * 1e6:.0f} m²)"


def _class_layers(lcm: LandCoverMap, prefix: str) -> list[Layer]:
    return [Layer(name=f"{prefix}: {CLASS_LABELS[c]}", kind="mask", array=lcm.masks[c], color=CLASS_COLORS[c],
                  legend=CLASS_LABELS[c]) for c in CLASSES if lcm.masks[c].any()]


def _lcm_facts(lcm: LandCoverMap, r: Raster) -> dict[str, Any]:
    fr = lcm.fractions()
    facts: dict[str, Any] = {
        "method": lcm.method,
        "class_fraction_pct": {c: round(100 * v, 2) for c, v in fr.items()},
        "class_confidence": {c: round(lcm.class_confidence(c), 3) for c in CLASSES},
        "analysed_pixels_pct": round(100 * float(lcm.valid.mean()), 2),
    }
    if lcm.cloud is not None:
        facts["cloud_pct"] = round(100 * float(lcm.cloud.mean()), 2)
    if r.pixel_area_m2:
        h, w = r.shape
        facts["class_area_km2"] = {c: round(v * lcm.valid.sum() * r.pixel_area_m2 / 1e6, 4) for c, v in fr.items()}
    facts.update({k: v for k, v in lcm.params.items()})
    return facts


def best_landcover(ctx: ToolContext) -> tuple[LandCoverMap, Raster, str]:
    """Most informative land-cover map available: fusion > optical > SAR (computed if needed)."""
    for name in ("optical_sar_fusion", "landcover_optical", "landcover_sar"):
        res = ctx.upstream.get(name)
        if res is not None and res.ok and "lcm" in res.data:
            return res.data["lcm"], res.data["raster"], name
    opt = _optical(ctx)
    if opt is not None:
        return classify_optical(opt), opt, "landcover_optical (implicit)"
    sar = _sar(ctx)
    return classify_sar(sar), sar, "landcover_sar (implicit)"


# =================================================================== optical
class LandcoverOptical(Tool):
    spec = ToolSpec(
        name="landcover_optical",
        title="Spectral land-cover mapper",
        description="Maps water, vegetation, built-up and bare soil from optical/multispectral bands using "
                    "NDVI/MNDWI/NDBI/BSI (Sentinel-2) or colour heuristics (RGB). Detects clouds. Produces masks, "
                    "class fractions and per-class confidence.",
        tasks=(TaskType.VQA, TaskType.CAPTION, TaskType.GROUNDING, TaskType.CROSS_MODAL),
        input_configs=SINGLE_OR_PAIR,
        kind="algorithmic",
        params=(ParamSpec("min_region_px", "int", 20, "Smallest connected region reported", 1, 5000),),
    )

    def run(self, ctx: ToolContext, params: dict[str, Any]) -> ToolResult:
        r = _optical(ctx)
        if r is None:
            return ToolResult(self.spec.name, "unsupported", notes=["No optical/multispectral image in the input."])
        lcm = classify_optical(r)
        facts = _lcm_facts(lcm, r)
        top = max(CLASSES, key=lambda c: facts["class_fraction_pct"][c])
        layers = _class_layers(lcm, "optical")
        if "NDVI" in lcm.indices:
            layers.append(Layer("optical: NDVI", "heatmap", lcm.indices["NDVI"], cmap="RdYlGn", vmin=-0.2, vmax=0.9,
                                legend="NDVI (-0.2 .. 0.9)"))
        if lcm.cloud is not None and lcm.cloud.any():
            layers.append(Layer("optical: cloud", "mask", lcm.cloud, color=CLASS_COLORS["cloud"], legend="cloud"))
        conf = float(np.mean([lcm.class_confidence(c) for c in CLASSES]))
        ans = (f"Optical analysis ({lcm.method}): dominant cover is {CLASS_LABELS[top]} "
               f"({facts['class_fraction_pct'][top]:.1f}%).")
        return ToolResult(self.spec.name, answer=ans, facts=facts, layers=layers, confidence=conf,
                          data={"lcm": lcm, "raster": r})


# =================================================================== SAR
class LandcoverSAR(Tool):
    spec = ToolSpec(
        name="landcover_sar",
        title="SAR backscatter land-cover mapper",
        description="Maps open water (low VV, adaptive Otsu threshold), built-up (double-bounce, high VV) and "
                    "vegetation (volume scattering, high VH) from Sentinel-1 VV/VH after Lee speckle filtering. "
                    "Works through cloud and at night.",
        tasks=(TaskType.VQA, TaskType.CAPTION, TaskType.GROUNDING, TaskType.CROSS_MODAL),
        input_configs=SINGLE_OR_PAIR,
        kind="algorithmic",
        params=(
            ParamSpec("water_threshold_db", "float", None, "Fixed VV water threshold in dB (default: automatic Otsu)",
                      -25.0, -10.0),
            ParamSpec("built_threshold_db", "float", -7.0, "VV threshold for double-bounce built-up", -12.0, 0.0),
            ParamSpec("speckle_window", "int", 5, "Lee filter window (odd)", 3, 11),
        ),
    )

    def run(self, ctx: ToolContext, params: dict[str, Any]) -> ToolResult:
        r = _sar(ctx)
        if r is None:
            return ToolResult(self.spec.name, "unsupported", notes=["No SAR image in the input."])
        win = int(params.get("speckle_window", 5)) | 1
        lcm = classify_sar(r, water_db=params.get("water_threshold_db"),
                           built_db=float(params.get("built_threshold_db", -7.0)), speckle_window=win)
        facts = _lcm_facts(lcm, r)
        layers = _class_layers(lcm, "SAR")
        layers.append(Layer("SAR: VV (filtered, dB)", "heatmap", lcm.indices["VV_filtered_dB"], cmap="gray",
                            vmin=-25, vmax=0, legend="VV dB (-25 .. 0)"))
        top = max(CLASSES, key=lambda c: facts["class_fraction_pct"][c])
        conf = float(np.mean([lcm.class_confidence(c) for c in CLASSES]))
        ans = (f"SAR analysis ({lcm.method}): water {facts['class_fraction_pct']['water']:.1f}%, built-up "
               f"{facts['class_fraction_pct']['built_up']:.1f}%, dominant {CLASS_LABELS[top]}.")
        return ToolResult(self.spec.name, answer=ans, facts=facts, layers=layers, confidence=conf,
                          data={"lcm": lcm, "raster": r})


# =================================================================== fusion
# Sensor weights per class: how much each modality is trusted for that class.
_W_OPT = {"water": 1.0, "vegetation": 1.2, "built_up": 0.8, "bare_soil": 1.0}
_W_SAR = {"water": 1.0, "vegetation": 0.6, "built_up": 1.1, "bare_soil": 0.5}


class OpticalSARFusion(Tool):
    spec = ToolSpec(
        name="optical_sar_fusion",
        title="Optical-SAR decision fusion",
        description="Fuses co-registered optical and SAR land-cover evidence with class-specific sensor weights; "
                    "fills cloud-covered optical pixels from SAR; reports per-class cross-sensor agreement and "
                    "where the sensors disagree.",
        tasks=(TaskType.CROSS_MODAL, TaskType.VQA, TaskType.GROUNDING),
        input_configs=(InputConfig.CROSS_MODAL,),
        kind="algorithmic",
        requires=("landcover_optical", "landcover_sar"),
        params=(ParamSpec("sar_weight", "float", 1.0, "Global multiplier on SAR evidence", 0.0, 3.0),),
    )

    def run(self, ctx: ToolContext, params: dict[str, Any]) -> ToolResult:
        opt_r, sar_r = _optical(ctx), _sar(ctx)
        if opt_r is None or sar_r is None:
            return ToolResult(self.spec.name, "unsupported", notes=["Fusion needs one optical and one SAR image."])
        up_o, up_s = ctx.upstream.get("landcover_optical"), ctx.upstream.get("landcover_sar")
        lo: LandCoverMap = up_o.data["lcm"] if up_o and up_o.ok else classify_optical(opt_r)
        ls: LandCoverMap = up_s.data["lcm"] if up_s and up_s.ok else classify_sar(sar_r)
        k = float(params.get("sar_weight", 1.0))
        h, w = lo.valid.shape
        cloud = lo.cloud if lo.cloud is not None else np.zeros((h, w), bool)
        opt_ok = lo.valid & ~cloud
        sar_ok = ls.valid
        scores = np.zeros((len(CLASSES), h, w), np.float32)
        for i, c in enumerate(CLASSES):
            scores[i] += np.where(lo.masks[c] & opt_ok, lo.confidence * _W_OPT[c], 0)
            scores[i] += np.where(ls.masks[c] & sar_ok, ls.confidence * _W_SAR[c] * k, 0)
        valid = opt_ok | sar_ok
        win = np.argmax(scores, axis=0)
        masks = {c: (win == i) & valid & (scores.max(axis=0) > 0) for i, c in enumerate(CLASSES)}
        opt_lab, sar_lab = lo.label_map(), ls.label_map()
        both = opt_ok & sar_ok
        agree = both & (opt_lab == sar_lab) & (opt_lab >= 0)
        disagree = both & (opt_lab != sar_lab)
        conf = np.zeros((h, w), np.float32)
        c_agree = 1 - (1 - lo.confidence) * (1 - ls.confidence)
        total = scores.sum(axis=0) + 1e-6
        c_dis = 0.8 * scores.max(axis=0) / total * np.maximum(lo.confidence, ls.confidence)
        conf = np.where(agree, c_agree, c_dis)
        conf = np.where(cloud & sar_ok, ls.confidence * 0.9, conf)  # SAR-only where cloudy
        fused = LandCoverMap(masks=masks, confidence=conf.astype(np.float32), valid=valid,
                             method="class-weighted optical+SAR decision fusion",
                             reliability={c: max(lo.reliability.get(c, .5), ls.reliability.get(c, .5)) for c in CLASSES})
        fr = fused.fractions()
        per_class = {}
        for c in CLASSES:
            a, b = lo.masks[c] & both, ls.masks[c] & both
            union = (a | b).sum()
            per_class[c] = {"optical_pct": round(100 * float(a.sum()) / max(both.sum(), 1), 2),
                            "sar_pct": round(100 * float(b.sum()) / max(both.sum(), 1), 2),
                            "agreement_iou": round(float((a & b).sum()) / union, 3) if union else None}
        cloud_filled = float((cloud & sar_ok).sum()) / (h * w)
        facts = {
            "method": fused.method,
            "fused_class_fraction_pct": {c: round(100 * v, 2) for c, v in fr.items()},
            "fused_class_confidence": {c: round(fused.class_confidence(c), 3) for c in CLASSES},
            "pixel_agreement_pct": round(100 * float(agree.sum()) / max(both.sum(), 1), 2),
            "per_class_cross_sensor": per_class,
            "cloud_pixels_filled_by_sar_pct": round(100 * cloud_filled, 2),
            "sensor_weights": {"optical": _W_OPT, "sar": {c: round(v * k, 2) for c, v in _W_SAR.items()}},
        }
        insights = []
        if cloud_filled > 0.005:
            insights.append(f"SAR provided the only usable observation for {_pct(cloud_filled)} of the scene "
                            f"hidden by cloud in the optical image.")
        for c in ("water", "built_up"):
            iou = per_class[c]["agreement_iou"]
            if iou is not None:
                insights.append(f"{CLASS_LABELS[c].capitalize()}: optical {per_class[c]['optical_pct']:.1f}% vs SAR "
                                f"{per_class[c]['sar_pct']:.1f}% (agreement IoU {iou:.2f}).")
        facts["insights"] = insights
        layers = _class_layers(fused, "fused")
        layers.append(Layer("fusion: sensor disagreement", "mask", disagree, color=(255, 0, 255),
                            legend="optical and SAR disagree"))
        layers.append(Layer("fusion: confidence", "heatmap", conf, cmap="viridis", vmin=0, vmax=1,
                            legend="fused confidence (0 .. 1)"))
        ans = ("Fused optical+SAR land cover: " + ", ".join(
            f"{CLASS_LABELS[c]} {facts['fused_class_fraction_pct'][c]:.1f}%" for c in CLASSES) +
               f". Sensors agree on {facts['pixel_agreement_pct']:.0f}% of jointly observed pixels.")
        return ToolResult(self.spec.name, answer=ans, facts=facts, layers=layers,
                          confidence=float(np.mean(conf[valid])) if valid.any() else 0.0,
                          data={"lcm": fused, "raster": opt_r})


# =================================================================== grounding
_SELECTIONS = ("auto", "largest", "smallest", "all", "north", "south", "east", "west", "centre")


class RegionGrounder(Tool):
    spec = ToolSpec(
        name="region_grounder",
        title="Land-cover region grounder",
        description="Text-guided grounding for land-cover referring expressions (e.g. 'the water body', 'the "
                    "largest built-up area in the north'). Selects connected regions of the requested class from "
                    "the best available land-cover map and returns boxes + a highlight mask.",
        tasks=(TaskType.GROUNDING, TaskType.CROSS_MODAL),
        input_configs=SINGLE_OR_PAIR,
        kind="algorithmic",
        params=(
            ParamSpec("target", "enum", "auto", "Class to ground (auto = parse from query)",
                      choices=("auto",) + CLASSES),
            ParamSpec("selection", "enum", "auto", "Which regions to return", choices=_SELECTIONS),
            ParamSpec("min_region_px", "int", 0, "Smallest region (0 = 0.1% of image, min 8 px)", 0, 100000),
        ),
    )

    def run(self, ctx: ToolContext, params: dict[str, Any]) -> ToolResult:
        target = params.get("target", "auto")
        if target == "auto":
            targets = [c for c in find_classes(ctx.query) if c in CLASSES][:3]
            if not targets:
                return ToolResult(self.spec.name, "unsupported",
                                  notes=["Query does not refer to a land-cover class this tool can ground "
                                         "(water, vegetation, built-up, bare soil)."])
        else:
            targets = [target]
        lcm, r, src = best_landcover(ctx)
        h, w = r.shape
        min_px = int(params.get("min_region_px") or max(8, int(0.001 * h * w)))
        sel = params.get("selection", "auto")
        if sel == "auto":
            quals = find_qualifiers(ctx.query)
            sel = next((q for q in quals if q in _SELECTIONS), None)
            plural = len(targets) > 1 or bool(re.search(
                r"\b(all|every|bodies|areas|regions|patches|fields|buildings|rivers|lakes|ponds)\b", ctx.query.lower()))
            sel = sel or ("all" if plural else "largest")
        from cv2 import connectedComponents
        boxes: list[Box] = []
        layers: list[Layer] = []
        per_target: list[dict[str, Any]] = []
        sentences: list[str] = []
        for t in targets:
            mask = clean_mask(lcm.masks[t], 1)
            regs = regions(mask, min_px=min_px, label=CLASS_LABELS[t])
            if not regs:
                sentences.append(f"no {CLASS_LABELS[t]} region was found")
                per_target.append({"target": t, "regions_found": 0})
                continue
            chosen = _select(regs, sel, h, w)
            _, lab = connectedComponents(mask.astype(np.uint8), connectivity=8)
            hl = np.isin(lab, [rg["id"] for rg in chosen])
            for rg in chosen:
                boxes.append(Box(rg["x0"], rg["y0"], rg["x1"], rg["y1"], label=CLASS_LABELS[t],
                                 score=float(np.mean(lcm.confidence[lab == rg["id"]])), source=self.spec.name))
            layers.append(Layer(f"grounded: {CLASS_LABELS[t]}", "mask", hl,
                                color=(255, 230, 0) if len(targets) == 1 else CLASS_COLORS[t],
                                legend=f"grounded {CLASS_LABELS[t]}"))
            desc = ", ".join(f"{_area_text(rg['area_frac'], r)} in the {rg['direction']}" for rg in chosen[:3])
            sentences.append(f"{len(chosen)} {CLASS_LABELS[t]} region(s) of {len(regs)} found "
                             f"({desc}{', ...' if len(chosen) > 3 else ''})")
            per_target.append({"target": t, "regions_found": len(regs), "regions_returned": len(chosen),
                               "covered_pct": round(100 * float(hl.mean()), 2),
                               "regions": [{k: (round(v, 4) if isinstance(v, float) else v)
                                            for k, v in rg.items() if k != "id"} for rg in chosen[:8]]})
        facts = {"targets": targets, "selection": sel, "landcover_source": src, "min_region_px": min_px,
                 "per_target": per_target}
        if not boxes:
            return ToolResult(self.spec.name, answer="Nothing to highlight: " + "; ".join(sentences) + ".",
                              facts=facts, confidence=float(np.mean([lcm.reliability.get(t, .5) for t in targets])) * .8)
        conf = float(np.mean([b.score for b in boxes])) * (0.95 if sel != "all" or len(boxes) < 4 else 0.9)
        ans = f"Highlighted [{sel}] " + "; ".join(sentences) + "."
        return ToolResult(self.spec.name, answer=ans, facts=facts, boxes=boxes, confidence=conf, layers=layers)


def _select(regs: list[dict], sel: str, h: int, w: int) -> list[dict]:
    if sel == "all":
        return regs
    if sel == "smallest":
        return [regs[-1]]
    if sel in ("north", "south", "east", "west", "centre"):
        key = {"north": lambda r: r["cy"], "south": lambda r: -r["cy"], "west": lambda r: r["cx"],
               "east": lambda r: -r["cx"],
               "centre": lambda r: (r["cy"] - h / 2) ** 2 + (r["cx"] - w / 2) ** 2}[sel]
        big = [r for r in regs if r["area_px"] >= 0.2 * regs[0]["area_px"]] or regs
        return [min(big, key=key)]
    return [regs[0]]


# =================================================================== reasoning
_PRESENCE = re.compile(r"\b(is there|are there|is any|are any|any|present|visible|exist|contain|contains|can you see|do you see|does .* have)\b")
_AREA = re.compile(r"\b(how much|what percentage|percent|percentage|fraction|proportion|share|area of|how large|coverage|cover)\b")
_COMPARE = re.compile(r"\b(more|larger|bigger|greater|less|fewer|smaller|than|compared|exceed)\b")
_DOMINANT = re.compile(r"\b(dominant|dominates|main|majority|most|primary|predominant|prevailing|mostly)\b")
_URBAN = re.compile(r"\b(rural or urban|urban or rural|is (this|it|the area) (a )?(rural|urban))\b")
_COUNT = re.compile(r"\b(how many|number of|count)\b")
_CAPTION = re.compile(r"\b(describe|description|caption|summar|overview|what (is|can be) (seen|visible|in))\b")

PRESENCE_MIN_FRAC = 0.005


class LandcoverReasoner(Tool):
    spec = ToolSpec(
        name="landcover_reasoner",
        title="Land-cover evidence reasoner",
        description="Answers presence / area / comparison / dominant-class / rural-urban / count questions and "
                    "writes a land-cover scene description from the best available land-cover evidence "
                    "(fusion > optical > SAR). Returns 'unsupported' for questions outside land cover.",
        tasks=(TaskType.VQA, TaskType.CAPTION, TaskType.CROSS_MODAL),
        input_configs=SINGLE_OR_PAIR,
        kind="reasoning",
        params=(ParamSpec("mode", "enum", "auto", "qa | describe | auto", choices=("auto", "qa", "describe")),),
    )

    def run(self, ctx: ToolContext, params: dict[str, Any]) -> ToolResult:
        lcm, r, src = best_landcover(ctx)
        fr = lcm.fractions()
        q = ctx.query.lower()
        mode = params.get("mode", "auto")
        if mode == "auto":
            mode = "describe" if _CAPTION.search(q) and not q.strip().startswith(("is ", "are ", "how ")) else "qa"
        base_facts = {"landcover_source": src, "class_fraction_pct": {c: round(100 * v, 2) for c, v in fr.items()}}
        if mode == "describe":
            return self._describe(lcm, r, src, base_facts)
        classes = [c for c in find_classes(q) if c in CLASSES]
        h, w = r.shape

        def conf_for(c: str, margin: float) -> float:
            return float(np.clip(lcm.class_confidence(c) * (0.7 + 0.3 * min(1.0, margin)), 0.05, 0.98))

        if _URBAN.search(q):
            bu = fr["built_up"]
            ans = "urban" if bu > 0.15 else "rural"
            return self._res(ans, f"{ans.capitalize()} (built-up covers {_pct(bu)}; threshold 15%).",
                             conf_for("built_up", abs(bu - 0.15) / 0.15), base_facts, "rural_urban")
        if _COUNT.search(q) and classes:
            c = classes[0]
            n = len(regions(clean_mask(lcm.masks[c], 1), min_px=max(8, int(0.001 * h * w))))
            return self._res(str(n), f"{n} distinct {CLASS_LABELS[c]} region(s) (connected components ≥0.1% of "
                                     f"the image).", conf_for(c, 0.3) * 0.7, base_facts, "count",
                             note="Counts land-cover patches, not individual objects.")
        if re.search(r"\b(equal|same (number|amount))\b", q):
            if len(classes) >= 2:
                a, b = classes[0], classes[1]
                same = abs(fr[a] - fr[b]) <= 0.1 * max(fr[a], fr[b], 1e-6)
                ans = "yes" if same else "no"
                return self._res(ans, f"{ans.capitalize()}: {CLASS_LABELS[a]} {_pct(fr[a])} vs {CLASS_LABELS[b]} "
                                      f"{_pct(fr[b])} (equal within 10%).", min(conf_for(a, 0.5), conf_for(b, 0.5)),
                                 base_facts, "equality")
            return ToolResult(self.spec.name, "unsupported", facts=base_facts,
                              notes=["Both objects map to the same land-cover class; equality of object counts "
                                     "needs object-level detection."])
        if re.search(r"\bthan\b", q) and len(classes) < 2:
            return ToolResult(self.spec.name, "unsupported", facts=base_facts,
                              notes=["The compared objects map to the same land-cover class; comparing their counts "
                                     "needs object-level detection."])
        if _COMPARE.search(q) and len(classes) >= 2:
            a, b = classes[0], classes[1]
            less = bool(re.search(r"\b(less|fewer|smaller)\b", q))
            res = fr[a] < fr[b] if less else fr[a] > fr[b]
            margin = abs(fr[a] - fr[b]) / max(fr[a], fr[b], 1e-6)
            ans = "yes" if res else "no"
            return self._res(ans, f"{ans.capitalize()}: {CLASS_LABELS[a]} {_pct(fr[a])} vs {CLASS_LABELS[b]} "
                                  f"{_pct(fr[b])}.", min(conf_for(a, margin), conf_for(b, margin)), base_facts,
                             "comparison")
        if _DOMINANT.search(q) and not classes or (_DOMINANT.search(q) and "land" in q):
            top = max(CLASSES, key=lambda c: fr[c])
            second = sorted(fr.values())[-2]
            return self._res(CLASS_LABELS[top], f"The dominant land cover is {CLASS_LABELS[top]} ({_pct(fr[top])}).",
                             conf_for(top, (fr[top] - second) / max(fr[top], 1e-6)), base_facts, "dominant")
        if _AREA.search(q) and classes:
            c = classes[0]
            return self._res(f"{100 * fr[c]:.1f}%", f"{CLASS_LABELS[c].capitalize()} covers {_area_text(fr[c], r)} "
                                                    f"of the analysed area.", conf_for(c, 0.5), base_facts, "area")
        if classes and (_PRESENCE.search(q) or q.strip().startswith(("is ", "are ", "does ", "do "))):
            c = classes[0]
            present = fr[c] >= PRESENCE_MIN_FRAC
            ans = "yes" if present else "no"
            margin = abs(fr[c] - PRESENCE_MIN_FRAC) / PRESENCE_MIN_FRAC
            where = spatial_distribution(lcm.masks[c]) if present else ""
            return self._res(ans, f"{ans.capitalize()}: {CLASS_LABELS[c]} covers {_pct(fr[c])}"
                                  f"{', ' + where if where else ''}.", conf_for(c, min(margin, 3) / 3), base_facts,
                             "presence")
        return ToolResult(self.spec.name, "unsupported", facts=base_facts,
                          notes=["Question type is outside land-cover presence/area/comparison/count; "
                                 "deferring to the vision-language model."])

    def _res(self, short: str, text: str, conf: float, facts: dict, qtype: str, note: str | None = None) -> ToolResult:
        f = {**facts, "question_type": qtype, "short_answer": short}
        return ToolResult(self.spec.name, answer=text, facts=f, confidence=conf, notes=[note] if note else [])

    def _describe(self, lcm: LandCoverMap, r: Raster, src: str, facts: dict) -> ToolResult:
        fr = lcm.fractions()
        order = sorted(CLASSES, key=lambda c: fr[c], reverse=True)
        parts = [f"{CLASS_LABELS[c]} ({_pct(fr[c])}, {spatial_distribution(lcm.masks[c])})"
                 for c in order if fr[c] >= 0.01]
        sensor = {"multispectral": "multispectral", "optical_rgb": "optical", "sar": "SAR"}[r.modality.value]
        if src.startswith("optical_sar_fusion"):
            sensor = "fused optical and SAR"
        text = f"Land-cover description from {sensor} evidence: the scene is dominated by {parts[0]}" if parts else \
            "No land-cover class could be mapped reliably."
        if len(parts) > 1:
            text += ", followed by " + "; ".join(parts[1:])
        text += "."
        if lcm.cloud is not None and lcm.cloud.mean() > 0.01:
            text += f" Clouds obscure {_pct(float(lcm.cloud.mean()))} of the image."
        conf = float(np.mean([lcm.class_confidence(c) for c in order[:2]]))
        return ToolResult(self.spec.name, answer=text, facts={**facts, "question_type": "describe"},
                          confidence=conf)
