import pytest

from satquery.demo import synth as S
from satquery.imagery.validate import validate_inputs
from satquery.tools.base import ParamSpec, ToolContext
from satquery.tools.landcover import classify_optical, classify_sar
from satquery.tools.registry import default_registry
from satquery.tools.vocab import find_classes

REG = default_registry()


def run_chain(query, rasters, names):
    rep = validate_inputs(rasters)
    ctx = ToolContext(query, rep.config, rep.rasters)
    for n in names:
        t = REG.get(n)
        ctx.upstream[n] = t.run(ctx, t.defaults())
    return ctx


def test_vocab_synonyms():
    assert find_classes("Highlight the water body near the town") == ["water", "built_up"]
    assert find_classes("forest and bare soil") == ["vegetation", "bare_soil"]


def test_param_spec_coercion_and_bounds():
    p = ParamSpec("s", "float", 1.0, "", 0.5, 2.0)
    assert p.coerce("5") == (2.0, "clipped to maximum 2.0")
    assert p.coerce("abc")[0] == 1.0
    e = ParamSpec("m", "enum", "auto", "", choices=("auto", "cva"))
    assert e.coerce("CVA") == ("cva", None)
    assert e.coerce("rm -rf")[0] == "auto"


@pytest.mark.parametrize("cls", ["water", "vegetation", "built_up"])
def test_optical_and_sar_fractions_close_to_truth(labels, s2, s1, cls):
    truth = S.truth_fractions(labels[0])[cls]
    assert abs(classify_optical(s2).fractions()[cls] - truth) < 0.03
    assert abs(classify_sar(s1).fractions()[cls] - truth) < 0.03


def test_fusion_fills_clouds_from_sar(labels, s2_cloudy, s1):
    ctx = run_chain("identify water", [s2_cloudy, s1], ["landcover_optical", "landcover_sar", "optical_sar_fusion"])
    f = ctx.upstream["optical_sar_fusion"]
    assert f.ok and f.facts["cloud_pixels_filled_by_sar_pct"] > 5
    truth = S.truth_fractions(labels[0])
    fused = f.facts["fused_class_fraction_pct"]
    # optical alone loses the clouded area; fusion restores near-true water fraction
    assert abs(fused["water"] / 100 - truth["water"]) < 0.03


def test_grounder_finds_largest_water_body(s2):
    ctx = run_chain("Highlight the water body referred to in the query.", [s2], ["landcover_optical", "region_grounder"])
    g = ctx.upstream["region_grounder"]
    assert g.ok and len(g.boxes) == 1 and g.boxes[0].label == "water"
    b = g.boxes[0]
    # lake is centred at (0.22 h, 0.25 w) of a 128 px scene
    assert b.x0 < 0.25 * 128 < b.x1 and b.y0 < 0.22 * 128 < b.y1


def test_grounder_unsupported_for_non_landcover_query(s2):
    ctx = run_chain("Highlight the airplanes.", [s2], ["region_grounder"])
    assert ctx.upstream["region_grounder"].status == "unsupported"


@pytest.mark.parametrize("q,short", [("Is there any water?", "yes"),
                                     ("Is there more vegetation than built-up area?", "yes"),
                                     ("Is there more water than vegetation?", "no"),
                                     ("Is this a rural or urban area?", "rural")])
def test_reasoner_answers(s2, q, short):
    ctx = run_chain(q, [s2], ["landcover_optical", "landcover_reasoner"])
    r = ctx.upstream["landcover_reasoner"]
    assert r.ok and r.facts["short_answer"] == short


def test_change_detection_matches_truth(labels, s2, s2_t2):
    ctx = run_chain("What changed?", [s2, s2_t2], ["change_detector", "change_semantics", "change_qa"])
    det = ctx.upstream["change_detector"]
    truth = (labels[0] != labels[1])
    m = det.data["mask"]
    iou = (m & truth).sum() / (m | truth).sum()
    assert iou > 0.7
    pc = ctx.upstream["change_semantics"].facts["per_class"]
    assert pc["built_up"]["direction"] == "increased" and pc["water"]["direction"] == "decreased"
    top = ctx.upstream["change_semantics"].facts["top_transitions"][0]
    assert (top["from"], top["to"]) == ("vegetation", "built_up")


def test_sar_change_detection(labels, s1, s1_t2):
    ctx = run_chain("Has the built-up area increased?", [s1, s1_t2], ["change_detector", "change_semantics", "change_qa"])
    assert ctx.upstream["change_qa"].facts["short_answer"] == "yes"


def test_unchanged_pair_reports_no_change(labels):
    a = S.render_s2(labels[0], seed=11)
    b = S.render_s2(labels[0], seed=12, gain=0.93)
    ctx = run_chain("Is there any change?", [a, b], ["change_detector", "change_semantics", "change_qa"])
    assert ctx.upstream["change_detector"].facts["changed_pct"] < 0.5
    assert ctx.upstream["change_qa"].facts["short_answer"] == "no"


@pytest.mark.parametrize("q,short", [("What did the water turn into?", "bare soil"),
                                     ("Has the built-up area decreased?", "no"),
                                     ("How much of the area changed?", "0%_to_10%")])
def test_change_qa_question_types(s2, s2_t2, q, short):
    ctx = run_chain(q, [s2, s2_t2], ["change_detector", "change_semantics", "change_qa"])
    assert ctx.upstream["change_qa"].facts["short_answer"] == short


# ---- flood questions and undated / screenshot pairs ------------------------------------------------------
def _rgb_copy(r, name, acquired=None, data=None):
    """The scene as a true-colour PNG without georeferencing or a capture date, like a screenshot."""
    import dataclasses
    rgb = r.data[[r.bands.index(b) for b in ("B04", "B03", "B02")]] if data is None else data
    return dataclasses.replace(r, data=rgb, bands=["R", "G", "B"], source_format="png", crs=None, transform=None,
                               bounds=None, name=name, acquired=acquired)


def test_flood_question_on_sentinel_pair_needs_new_water(s2, s2_t2):
    # in the demo pair water shrinks (it becomes bare soil): measured with infrared bands, that is no flood
    ctx = run_chain("Has the area been flooded?", [s2, s2_t2], ["change_detector", "change_semantics", "change_qa"])
    qa = ctx.upstream["change_qa"]
    assert qa.facts["question_type"] == "flood" and qa.facts["short_answer"] == "no"


def test_flood_question_on_true_colour_pair_counts_grey_ground_and_says_so(s2):
    import cv2
    import numpy as np

    from satquery.tools.landcover import classify
    from satquery.tools.vocab import CLASSES
    before = _rgb_copy(s2, "before.png")
    m = classify(before).label_map()
    grey = before.data[:, m == CLASSES.index("built_up")].mean(axis=1).reshape(3, 1, 1)   # the scene's grey ground
    water = cv2.dilate((m == CLASSES.index("water")).astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    flooded = before.data.copy()
    flooded[:, water] = np.broadcast_to(grey, flooded.shape)[:, water]   # muddy deposits spread out from the river
    after = _rgb_copy(s2, "after.png", data=flooded)
    ctx = run_chain("Is there flooding?", [before, after], ["change_detector", "change_semantics", "change_qa"])
    qa = ctx.upstream["change_qa"]
    assert qa.facts["short_answer"] == "yes" and "true-colour" in qa.answer
    river_conf = qa.confidence
    assert "the first image and the second image" in ctx.upstream["change_detector"].answer   # no fake dates
    # control: the same amount of new grey ground as separate building-sized patches inside vegetation, away from
    # any water or grey ground, must not get the river-bed shape support
    other = cv2.medianBlur((m != CLASSES.index("vegetation")).astype(np.uint8), 5)   # smoothed: ignore speckle
    inner = ~cv2.dilate(other, np.ones((15, 15), np.uint8)).astype(bool)
    patches = before.data.copy()
    placed = 0
    for y, x in zip(*np.nonzero(inner)):
        if placed < 12 and inner[y, x] and 4 <= y < 124 and 4 <= x < 124:
            patches[:, y - 3:y + 4, x - 3:x + 4] = grey
            inner[max(0, y - 14):y + 15, max(0, x - 14):x + 15] = False   # keep the patches apart
            placed += 1
    ctx = run_chain("Is there flooding?", [before, _rgb_copy(s2, "patches.png", data=patches)],
                    ["change_detector", "change_semantics", "change_qa"])
    patch_qa = ctx.upstream["change_qa"]
    assert placed >= 8 and "river bed" not in patch_qa.answer
    assert patch_qa.facts["short_answer"] == "no" or patch_qa.confidence < river_conf
    same = _rgb_copy(s2, "before_again.png")
    ctx = run_chain("Is there flooding?", [before, same], ["change_detector", "change_semantics", "change_qa"])
    assert ctx.upstream["change_qa"].facts["short_answer"] == "no"   # nothing changed: no flood


def test_cdvqa_specialist_steps_aside_on_flood_questions(s2, s2_t2):
    from satquery.tools.cdvqa import CDVQASpecialist
    tool = CDVQASpecialist()
    if not tool.available()[0]:
        pytest.skip("CDVQA specialist not trained")
    rep = validate_inputs([_rgb_copy(s2, "a.png"), _rgb_copy(s2_t2, "b.png")])
    res = tool.run(ToolContext("Has the area been flooded?", rep.config, rep.rasters), tool.defaults())
    assert res.status == "unsupported"
