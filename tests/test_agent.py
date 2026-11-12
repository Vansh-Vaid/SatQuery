"""Controller tests: routing per input configuration, plan validation, VLM integration, Claude planner."""
import json
from types import SimpleNamespace

from satquery.agent.controller import SatQueryAgent
from satquery.agent.plan import Plan, PlanStep, validate_plan
from satquery.agent.planner import ClaudePlanner, RulePlanner
from satquery.imagery.validate import InputConfig, validate_inputs
from satquery.tools.base import TaskType
from satquery.tools.registry import default_registry
from satquery.tools.vlm import RSVLM


class FakeVLM:
    name = "fake-vlm"

    def __init__(self, text="Yes, there is a lake in the north-west.", conf=0.9):
        self.text, self.conf, self.calls = text, conf, []

    def generate(self, images, prompt, max_new_tokens=128):
        self.calls.append((len(images), prompt))
        return self.text, self.conf


def agent(vlm=None):
    reg = default_registry(RSVLM(backend=vlm) if vlm else RSVLM(backend_factory=lambda: None))
    return SatQueryAgent(registry=reg, planner=RulePlanner())


def tools_run(resp):
    return [s["tool"] for s in resp.trace["steps"]]


def test_routes_each_input_configuration(s2, s1, s2_t2, s2_cloudy):
    a = agent()
    r = a.run("Describe the land-cover and major objects visible in this image.", [s2])
    assert r.status == "ok" and r.task == "caption" and "landcover_reasoner" in tools_run(r)
    r = a.run("Highlight the water body referred to in the query.", [s2])
    assert r.task == "grounding" and r.boxes and r.boxes[0]["label"] == "water"
    r = a.run("What changed between these two dates, and where did the change occur?", [s2, s2_t2])
    assert r.task == "change_description" and tools_run(r) == ["change_detector", "change_semantics", "change_qa"]
    r = a.run("Use the optical and SAR images together to identify built-up and water-covered regions.", [s2_cloudy, s1])
    assert r.task == "cross_modal_analysis"
    assert {"landcover_optical", "landcover_sar", "optical_sar_fusion"} <= set(tools_run(r))
    r = a.run("Has the built-up area increased, decreased, or remained unchanged?", [s2, s2_t2])
    assert r.task == "change_vqa" and r.short_answer == "increased"


def test_trace_is_auditable(s2):
    r = agent().run("Is there any water?", [s2])
    t = r.trace
    assert {"request_id", "query", "input", "validation", "planner", "plan", "steps", "result"} <= set(t)
    for s in t["steps"]:
        assert {"tool", "params", "status", "duration_ms", "output"} <= set(s)
    json.dumps(t)  # must be serialisable


def test_rejects_invalid_input_without_running_tools(s2, s1, s2_t2):
    r = agent().run("anything", [s2, s1, s2_t2])
    assert r.status == "rejected" and "steps" not in r.trace


def test_vlm_used_and_cross_checked(s2):
    fake = FakeVLM()
    r = agent(fake).run("Is there any water?", [s2])
    assert tools_run(r)[0] == "rs_vlm" and fake.calls and fake.calls[0][0] == 1
    assert r.short_answer == "yes" and "confirmed" in " ".join(r.trace["result"]["integration_notes"])


def test_vlm_disagreement_lowers_confidence(s2):
    agree = agent(FakeVLM("Yes.", 0.9)).run("Is there any water?", [s2]).confidence
    dis = agent(FakeVLM("No.", 0.9)).run("Is there any water?", [s2]).confidence
    assert dis < agree


def test_vlm_grounding_boxes_verified(s2):
    fake = FakeVLM('[{"bbox_2d": [140, 90, 400, 340], "label": "lake"}]', 0.8)
    r = agent(fake).run("Highlight the water body referred to in the query.", [s2])
    vlm_boxes = [b for b in r.boxes if b["source"] == "rs_vlm"]
    assert vlm_boxes and "evidence" in vlm_boxes[0]["label"]


def test_cross_modal_vlm_gets_both_images(s2, s1):
    fake = FakeVLM("Built-up area in the east, water in the north-west.")
    agent(fake).run("Describe the scene using both sensors.", [s2, s1])
    assert fake.calls[0][0] == 2 and "Sentinel-1 SAR" in fake.calls[0][1]


def test_vlm_failure_degrades_gracefully(s2):
    class Broken(FakeVLM):
        def generate(self, *a, **k):
            raise RuntimeError("CUDA out of memory")
    r = agent(Broken()).run("Is there any water?", [s2])
    assert r.status == "ok" and r.short_answer == "yes"
    assert next(s for s in r.trace["steps"] if s["tool"] == "rs_vlm")["status"] == "error"


def test_validate_plan_corrections(s2):
    reg = default_registry(RSVLM(backend_factory=lambda: None))
    p = Plan(TaskType.CHANGE_VQA, "qa", [
        PlanStep("made_up_tool"), PlanStep("change_qa"), PlanStep("rs_vlm"),
        PlanStep("region_grounder", {"selection": "largest", "evil": "1"}),
        PlanStep("landcover_sar", {"speckle_window": "99"})], "test")
    v = validate_plan(p, reg, InputConfig.SINGLE)
    names = [s.tool for s in v.steps]
    joined = " | ".join(v.corrections)
    assert v.task is TaskType.VQA
    assert "made_up_tool" not in names and "change_qa" not in names and "rs_vlm" not in names
    assert "undeclared parameter 'evil'" in joined and "clipped to maximum 11" in joined
    assert v.steps[names.index("landcover_sar")].params["speckle_window"] == 11


def test_validate_plan_inserts_dependencies(s2, s2_t2):
    reg = default_registry(RSVLM(backend_factory=lambda: None))
    v = validate_plan(Plan(TaskType.CHANGE_VQA, "qa", [PlanStep("change_qa")], "t"), reg, InputConfig.BITEMPORAL)
    assert [s.tool for s in v.steps] == ["change_detector", "change_semantics", "change_qa"]


class FakeClient:
    """Mimics anthropic client.beta.messages.create returning structured JSON."""

    def __init__(self, payload, stop_reason="end_turn"):
        self.payload, self.stop_reason, self.kwargs = payload, stop_reason, None
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.kwargs = kw
        return SimpleNamespace(stop_reason=self.stop_reason,
                               content=[SimpleNamespace(type="text", text=json.dumps(self.payload))])


def test_claude_planner_plan_is_validated(s2, s2_t2):
    reg = default_registry(RSVLM(backend_factory=lambda: None))
    payload = {"task": "change_vqa", "intent": "qa", "rationale": "bi-temporal built-up question",
               "steps": [{"tool": "change_detector", "purpose": "map", "params": [{"name": "sensitivity", "value": "1.3"}]},
                         {"tool": "change_qa", "purpose": "answer", "params": []}]}
    client = FakeClient(payload)
    ag = SatQueryAgent(registry=reg, planner=ClaudePlanner(reg, client=client))
    r = ag.run("Has the built-up area increased?", [s2, s2_t2])
    assert r.trace["planner"]["name"].startswith("claude:")
    assert tools_run(r) == ["change_detector", "change_semantics", "change_qa"]  # dependency inserted
    assert r.trace["steps"][0]["params"]["sensitivity"] == 1.3
    sent = client.kwargs
    assert sent["model"] == "claude-opus-5-5" and sent["output_config"]["format"]["type"] == "json_schema"
    assert sent["extra_body"] == {"fallbacks": "default"}


def test_claude_planner_failure_falls_back_to_rules(s2):
    reg = default_registry(RSVLM(backend_factory=lambda: None))
    ag = SatQueryAgent(registry=reg, planner=ClaudePlanner(reg, client=FakeClient({}, stop_reason="refusal")))
    r = ag.run("Is there any water?", [s2])
    assert r.status == "ok" and "fell back to the rule planner" in " ".join(r.trace["plan"]["corrections"])


def test_vlm_explicit_mode_and_prompt_override_beat_the_pair_guess(s2, s2_t2):
    # benchmarks pass the training prompt and task; a two-image input must not turn grounding into "change"
    rep = validate_inputs([s2, s2_t2])
    assert rep.config is InputConfig.BITEMPORAL
    from satquery.tools.base import ToolContext
    fake = FakeVLM(text='[{"bbox_2d": [100, 200, 500, 600], "label": "lake"}]')
    res = RSVLM(backend=fake).run(ToolContext("Locate: the lake", rep.config, rep.rasters),
                                 {"mode": "grounding", "prompt": "EXACT TRAINING PROMPT"})
    assert fake.calls[-1][1] == "EXACT TRAINING PROMPT"
    assert res.facts["task"] == "grounding" and len(res.boxes) == 1
    # without an explicit mode a two-date pair is still read as a change question
    RSVLM(backend=fake).run(ToolContext("What changed?", rep.config, rep.rasters), {"mode": "auto"})
    assert "Compare Image 1" in fake.calls[-1][1]
