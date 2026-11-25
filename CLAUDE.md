# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

SatQuery answers plain-English questions about satellite images (one image, an optical + radar pair, or a before/after pair) with an answer, map evidence, a confidence score and an execution trace. FastAPI + PyTorch backend, React/three.js frontend, runs on one CPU container.

## Commands

```bash
pip install -r requirements.txt          # app runtime (requirements-train.txt for training)
python run.py                            # app on http://localhost:8000 (/console is the analysis UI)
python run.py --no-browser --port 8080
python run.py --setup-models             # download data, train the specialists (slow, ~17 GB traffic)
python run.py --setup-highres            # download the trained high-resolution models (~30 MB)

python -m pytest -q                      # unit tests (tests/ only, about 80 s)
python -m pytest -q tests/test_agent.py::test_trace_is_auditable
python -m pytest -q e2e                  # Playwright smoke test; needs playwright + chromium
python -m ruff check .                   # CI pins ruff 0.15.21; line length 120

cd frontend && npm ci
npm run dev                              # Vite; proxies /api to a backend on :8000
npm run build                            # tsc --noEmit, then writes ../satquery/web/dist

python scripts/run_query.py "Is there water?" demo_data/s2_2024-06-01.tif   # one query, no server
python eval/run_eval.py --data eval/data/<set>.jsonl --mode agent --out eval/results/<name>
                                         # --mode vlm scores rs_vlm alone
```

Model tests skip when their checkpoint is missing, so a green run without `checkpoints/` covers less than CI does.

## Request flow

`SatQueryAgent.run` (`satquery/agent/controller.py`) is the spine; the API, CLI and eval runner all call it.

1. **Validate** (`imagery/validate.py`): chooses the `InputConfig` (single / cross_modal / bitemporal), orders dates, reprojects to a common grid. Failures return `rejected` without running tools.
2. **Plan** (`agent/planner.py`): `ClaudePlanner` (structured JSON) or the deterministic `RulePlanner`. `SATQUERY_PLANNER=auto` uses Claude whenever credentials are found (`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` or `~/.config/anthropic`), which costs money. A `PlannerError` falls back to rules.
3. **Gate** (`agent/plan.py: validate_plan`): every plan passes through it. Drops unknown, unavailable or incompatible tools, inserts `requires` dependencies, rejects undeclared params and clips values. Every correction goes in the trace.
4. **Execute**: tools run in order and share results through `ToolContext.upstream` (keyed by tool name). A tool exception becomes an `error` result, never a failed request.
5. **Integrate** (`_integrate`): picks one primary tool per task, cross-checks it against measured evidence, and adjusts confidence (agreement raises it, disagreement x0.7, each input warning costs 4%). **Tool precedence lives here, not in the planner.** When `rs_vlm` is available it takes precedence for single-image VQA and captions, so enabling a weak VLM can lower accuracy.

Image units are fixed at load (`imagery/raster.py`): optical is reflectance in [0, 1], SAR is decibels. Tools never re-guess units.

## Adding or changing a tool

- Subclass `Tool` with a `ToolSpec` (`tools/base.py`) and register it in `default_registry()` (`tools/registry.py`). Then wire it into `RulePlanner.plan` **and** `_integrate`; registration alone only exposes it to Claude.
- Tools abstain rather than guess: return `status="unsupported"` outside their domain, and make `available()` return false without a checkpoint. The app must work with no checkpoints at all.
- Domain routing is by pixel size: `hr_vqa`, `hr_grounder` and `cdvqa_vqa` abstain above 5 m/pixel, and `rsvqa_vqa` abstains below 3 m. Images of unknown scale (ungeoreferenced PNGs) are treated as high-resolution aerial.
- `ToolResult.facts` must be JSON-serialisable (it goes in the trace); pass arrays to later tools through `data`.
- Change events (flood, fire, urban growth...) are vocabulary in `tools/events.py` over one land-cover model (`change_mapper`). Do not add a model per event (decision D20).

## Models

- A trained model ships only if it beats the current behaviour on held-out data (its "gate"). Status of pending models: `docs/GENERALIZATION_PLAN.md`. Re-check calibration (`eval/calibration_report.py`) when tools or integration rules change.
- Compare models on the same rows: `eval/data/vrsbench_sample_*.jsonl` are stratified samples, so their overall accuracy is not comparable with a differently sampled set.
- VLM training, scoring and serving must share prompts (`models/vlm_prompts.py: build_prompt`) and `processor_kwargs` (`models/vlm_backend.py`). Change them only together with a retrain.
- `kaggle/*.ipynb` clone `main` from GitHub, so a fix under `training/` or `eval/` must be pushed before the next Kaggle run; changes to a notebook itself must be re-uploaded to Kaggle.
- `checkpoints/`, `data/`, `runs/` and `eval/results/` are not in git. CI fetches `*.pt` and `*.json` from the GitHub release `models-v1`; a new or retrained model must be uploaded there to reach production.

## Frontend build

`satquery/web/dist` is committed and served by FastAPI. After any change in `frontend/src`, run `npm run build` and commit the result; CI fails if the committed build differs from the source. On Windows the dist files can show as modified with an empty `git diff` (line endings only); do not commit those.

## Deployment

Pushing app paths to `main` runs the tests and e2e, then publishes `ghcr.io/vansh-vaid/satquery` (`.github/workflows/build-image.yml`). Going live is a separate manual step: `python scripts/deploy_azure.py`. The container has 1 CPU and 2 GB, and one process holds the rate limiter, demo cache and report store in memory, so do not add workers or replicas. Free-only: no paid APIs, GPU hosting or extra replicas.

## Reference

- `docs/TECHNICAL.md`: all environment variables, evaluation commands and results. Its architecture diagram still says 9 tools; `tools/registry.py` is authoritative.
- `docs/DECISIONS.md`: numbered decisions (D1...) with rejected alternatives. Read the relevant one before reversing a design choice, and add one for significant changes.
- Tests and demos use synthetic Sentinel scenes with known ground truth (`satquery/demo/synth.py`, fixtures in `tests/conftest.py`).
