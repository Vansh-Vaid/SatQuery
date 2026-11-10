# Design decisions

Each entry: the decision, the alternatives considered, and why.

### D1. Hybrid registry: one adapted VLM + algorithmic specialists
*Alternatives:* a single VLM for everything; many task-specific deep models.
*Why:* one LoRA run covers the adaptation requirement and free-form language; physically based tools
add pixel-level, quantitative, auditable evidence, work on CPU, and keep the system answering when the
GPU model is offline. Two independent sources also enable cross-checking (D7).

### D2. Base VLM: Qwen3-VL-2B-Instruct + LoRA
*Alternatives:* InternVL3-1B (the BigEarthNet.txt paper's choice), larger 7-8B models.
*Why:* native multi-image input (needed for S2+S1 and t1/t2), native box-grounding output convention
(JSON `bbox_2d`, 0-1000), fits LoRA training on one 16 GB GPU, supported by `AutoModelForImageTextToText`.
The VLM interface is backend-agnostic, so swapping models only means changing `SATQUERY_VLM_BASE`.

### D3. SAR enters the VLM as a rendered false-colour image, not a new encoder branch
*Alternative:* dual ViT branches (RS-InternVL style).
*Why:* reuses the pretrained vision encoder and processor unchanged, so LoRA adaptation is cheap; the
fixed dB windows (VV −25…0, VH −32…−5, ratio 0…15) keep the rendering physically consistent across
images. The rendering code is shared by training and inference (`models/vlm_prompts.py`).

### D4. Training on a stratified subset of BigEarthNet.txt
*Why:* the paper's full fine-tune took ~2 days on 4×H200. 6k samples per task type (≈24k) gives a
feasible single-GPU run; the manually verified `bench` split is kept whole for evaluation.

*Verified against the real file (2026-11-09):*
- **Size and types:** `BigEarthNet.txt.parquet` has 9,553,962 rows. The types are `binary`, `mcq`,
  `captioning` and `bounding box`; the splits are `train`, `validation`, `test` and `bench`.
- **Columns:** all eight columns read by `prepare_bigearthnet_txt.py` exist.
- **Box answers:** always exactly one box, written `[a b, c d]` in normalised 0–1 coordinates, for example
  `[0.64 0.0, 1.0 0.71]`. The parser's "normalised" branch handles this. Prompts use `<point>(a, b)</point>`
  and `<ref>…</ref>`, with points in the same coordinate order as the boxes.
- **Still to confirm on the GPU machine:** whether each pair is (x, y) or (row, column). Overlay one box on
  its image before training; training is consistent either way, but the UI draws (x, y).

### D5. Claude as planner, constrained by structured outputs, gated by a validator
*Why:* language understanding of arbitrary queries is what an LLM is good at; the JSON schema's tool enum
is the registry, and `validate_plan` enforces input compatibility, availability, dependency order and
parameter bounds, so the planner cannot run anything outside the registry. Any API failure falls back
to the deterministic rule planner, and the trace records which planner ran and every correction.
The adaptation requirement applies to the visual component; the planner never sees pixels.

### D6. Change detection: normalised CVA / log-ratio with an Otsu threshold *and* a physical floor
*Why:* plain Otsu always splits a histogram, so an unchanged pair would show "change". The floor
(1.2σ for optical z-scored CVA plus a 0.03 reflectance minimum; 3 dB for SAR) makes "no change" a
reachable answer. A learned Siamese U-Net (LEVIR-CD) can replace it via a checkpoint.

### D7. Confidence = source confidence × agreement × input quality
VLM confidence is the geometric-mean token probability; algorithmic confidence comes from margins to
decision thresholds and class reliability per sensor. Agreement between VLM and evidence raises it,
disagreement multiplies by 0.7, each input warning costs 4% (floor 70%). Labels: high ≥ 0.75,
medium ≥ 0.5. These are heuristic scores, not calibrated probabilities; calibrating them on the
benchmark predictions (reliability diagrams from `eval/run_eval.py` output) is a follow-up.

### D8. Cross-modal fusion is decision-level with class-specific sensor weights
Optical is trusted more for vegetation, SAR more for built-up (double-bounce) and equally for water;
optical cloud pixels are filled from SAR. Decision-level fusion keeps each sensor's evidence inspectable
(per-class agreement IoU, disagreement map), which is the "complementary information" the brief asks for.

### D10. UI stack: React + TS + three/react-three-fiber; plain CSS instead of Tailwind/Framer/GSAP
The design spec recommends a stack; I adopted the parts that carry the experience: React/TS/Vite, the
Three.js scenes, and a token set as CSS custom properties. Lenis smooth scrolling was tried and removed
(D12).
I did not add Tailwind, Framer Motion or GSAP: every specified motion (tilt, flip, sweeps, glyph scramble,
beams, DAG edge flow) is a few lines of CSS or rAF, and each library would add bundle weight and a second
styling system for no visible gain.
Deck.gl/MapLibre and client-side `geotiff.js` were also skipped. The backend already parses GeoTIFFs
(rasterio) and renders physically scaled layers, so the canvas composites server-rendered RGBA layers on
the reference pixel grid. That keeps one rendering path for the UI, the reports and the VLM.
The reticle's lat/lon comes from linear interpolation inside the WGS84 footprint (exact enough for
scene-sized tiles; marked as approximate in code). GeoJSON export uses the exact affine + CRS transform.

### D11. Landing page: "survey atlas", not "orbital brutalism"
The first landing draft reused the console's blackletter and neon-cyan HUD language. User feedback was
that it looked "spooky", not professional, and too tight. The landing page now uses the palette of what
the product maps: paper, ink-green, terracotta (built-up), lake blue and vegetation green, with an
editorial serif for headlines and generous spacing. The 3D work moved from atmosphere to explanation:
a lit, draggable atlas globe, and an exploded stack of the real analysis layers.
The palette then moved to off-white, burnt orange and olive at the user's request, and the console adopted
the same system (tokens in `styles/tokens.css`). The landing page was cut to about two screens because
the console is the product. Detail on pipelines, evidence and evaluation lives there and in this
repository's docs, not in a long marketing page.

### D12. Performance budget: compositor-only motion, on-demand WebGL, lazy report rendering
Measured problems: the landing hero ran at 19 fps, and an analysis took 6 s. The causes were:
- a heavy globe render (pixel ratio 2, 128-segment sphere, 2048 px texture);
- per-path SVG scale animations and repaint-heavy gradient animations;
- a typing effect that re-rendered the whole page every 26 ms;
- Lenis scroll latency;
- `backdrop-filter` blur over an always-animating canvas in the console;
- 45 maximally compressed report PNGs encoded on every request, although the UI never shows them.

Fixes:
- capped pixel ratio, lighter geometry and texture;
- motion limited to transform and opacity;
- the typing prompt isolated in a memoised component;
- native scrolling;
- solid panels and an on-demand console canvas;
- fast PNG compression, with report images built only when a report is downloaded.

Result: 60 fps on both pages, and about 0.3 s per warm analysis with a 0.9 MB response.

### D13. Remote-sensing adaptation on CPU now, VLM fine-tune on GPU later
The mandatory adaptation requirement was not met at runtime: the BigEarthNet.txt LoRA needs a GPU. The brief
also allows "any open-source training data", so two adapted components were trained on this CPU:
- **EuroSAT ResNet-18.** Trained on the official splits: 95.7% test accuracy, and 93% on raw 13-band
  GeoTIFFs rendered by SatQuery, so the rendering adds little domain gap.
- **RSVQA-LR specialist.** Frozen EuroSAT features plus land-cover fractions, fused with a bag-of-words
  question encoding by elementwise product (Lobry et al. 2020): 84.6% on the official test split.

Routing rules:
- The specialist answers closed questions (yes/no, rural/urban, counts) and abstains on open ones.
- The EuroSAT classifier answers land-use questions.
- The land-cover reasoner stays as a measured cross-check: agreement raises confidence, disagreement lowers it.

The first integration lost about 11 points to routing ("Is a road present?" went to the Highway class),
which the per-tool error attribution exposed and the fix recovered.

### D14. Change questions: a CDVQA-trained specialist, with the measured change tools as cross-check
On real aerial pairs, the rule-based change pipeline scored 18.7% on CDVQA. Its yes/no answers were near
chance, and it could not express CDVQA's answers at all: the six SECOND classes, 10% ratio bins, and
"changed into what". The pixel measurements were sound, but the mapping from measurements to the
benchmark's semantics was missing.

`cdvqa_vqa` learns that mapping from:
- frozen EuroSAT ResNet-18 features of both dates;
- their per-tile differences;
- pixel change statistics;
- SatQuery's land-cover fractions.

It is trained on 437 pairs (15k questions) and scores 65.1% on a 6,000-question subset of test set 1. The
paper's end-to-end ResNet-18, trained on 1,600 pairs, scores 67.7%. The majority answer per question type
scores 51%.

Routing:
- It is primary for change questions on optical pairs. `change_qa` stays as a cross-check: agreement on
  yes/no raises confidence, disagreement lowers it.
- "Describe the change" requests, SAR pairs and the change map still come from the measured tools.
- It abstains on question forms CDVQA never contains. None of its 21,000 questions offer choices or start
  with anything but What, Have, Did or How. The test suite caught this: on "Has built-up increased,
  decreased or remained unchanged?" the specialist forced a yes/no and overrode the correct measured
  "increased". The rule changes none of the benchmark questions.

Data: the mirror (`ljx620/CDVQA`) duplicates both images into every question (about 80 GB), so
`eval/fetch_cdvqa.py` streams evenly spaced shards, keeps each pair once, retries broken streams and
resumes per shard.

### D15. High-resolution imagery: VRSBench-trained specialists, routed by pixel size
VRSBench is 0.5-1 m Google Earth imagery about individual objects. Every earlier model was trained on 10 m
Sentinel data. The baseline agent scored 23.8% on its VQA sample, 0% on grounding (only land-cover regions
could be boxed) and low caption overlap.

Three additions, all on frozen ImageNet ResNet-18 features with a 3×3 spatial grid. The grid matters
because position questions are common.
- **`hr_vqa`.** 300-answer classifier: 54.0% exact match on all 37,409 evaluation questions (agent sample
  58.0%).
- **`hr_grounder`.** Box regressor: 14.5% Acc@0.5. That is honest but weak: the median target covers 1.2%
  of the image, beyond what a 3×3 grid can localise. Its confidence is capped at 0.35, and every answer
  says the box is approximate.
- **Describe mode of `hr_vqa`.** It asks the model about the setting, main object, count and position, and
  composes a sentence from the confident answers only. No captions are borrowed from other images, which
  would be misleading in a product even if they scored well.

Routing is by pixel size:
- The high-resolution tools abstain above 5 m per pixel, and `rsvqa_vqa` abstains below 3 m. Unknown scale
  (an ungeoreferenced PNG) goes to the high-resolution tools.
- A 300-question RSVQA re-check scored 87.0%, so the earlier results are unaffected.
- In grounding, a named object (vehicle, ship, bridge...) takes priority over a land-cover word in the same
  expression ("the vehicle next to the green area").

Captions are scored on the agent's full answer (the caption), not the one-line UI summary. Short captions
would otherwise be penalised by BLEU's brevity rule; scoring the summary made the new captions look worse
than they are. The real fix for captions and small-object grounding is the vision-language model (D2).

### D16. Deployment: one CPU container, free Hugging Face Space as the default host
*Alternatives:*
- Render, Railway or Fly free tiers: 256-512 MB of memory, below the 615 MB the app uses once warmed up.
- Serverless: cold starts reload PyTorch and every model on each request.
- Hosting the VLM: needs a paid GPU.

*Why:* a free CPU Space has 2 vCPUs and 16 GB, builds the same Dockerfile that works on any server, and
gives an HTTPS link. All the trained specialists are small CPU models (157 MB in total), so the full
measured system runs there. The VLM stays optional behind `SATQUERY_VLM_BACKEND=remote`.

Public-traffic guards, each covered by `tests/test_deploy.py`:
- a per-IP rate limit;
- a bounded number of concurrent analyses, with a 503 "busy" response instead of an unbounded queue;
- upload size enforced while streaming, plus an early check on request-body size;
- a cap on question length;
- `nosniff` and referrer headers.

The landing page runs three demo analyses for every visitor. Demo answers do not depend on the visitor, so
they are cached, shared and pre-computed at startup. Without that, landing-page traffic alone would use up
the two CPUs.

Uploaded files are deleted after each request, and run traces are not written to disk in the image.

### D9. Synthetic Sentinel-1/2 scenes with known ground truth for tests and demos
They let tests assert correctness (fractions, change IoU, directions), not just "it runs", and make the
demo work offline. Real-data accuracy is measured by `eval/` on the named benchmarks.
