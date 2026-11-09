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

### D9. Synthetic Sentinel-1/2 scenes with known ground truth for tests and demos
They let tests assert correctness (fractions, change IoU, directions), not just "it runs", and make the
demo work offline. Real-data accuracy is measured by `eval/` on the named benchmarks.
