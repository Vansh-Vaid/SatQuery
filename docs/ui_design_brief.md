# UI/UX Design Specification: SatQuery AI

**Project Name:** SatQuery AI  
**Domain:** Multimodal Remote Sensing (Optical/Multispectral & SAR), Agentic Vision-Language Orchestration & Geospatial Intelligence  
**Aesthetic Direction:** Orbital Brutalism meets Telemetry Glassmorphism  
**Design Inspiration:** Awwwards Site of the Year Winners (Active Theory, NASA JPL Interactive, Palantir Foundry HUDs, Resn)

---

## 1. Executive Summary & Creative Direction

**SatQuery AI** transforms complex, specialist remote-sensing workflows—such as synthetic aperture radar (SAR) backscatter analysis, multispectral band fusion, and bi-temporal change detection—into an intuitive, cinematic orbital command center. Rather than overwhelming non-expert users with rigid GIS toolbars, the user is placed inside an **Agentic Orbital Reconnaissance HUD**.

The centerpiece of the experience is an interactive **3D WebGL Earth & Sensor Constellation Canvas** set against a high-energy **Singularity / Deep Space backdrop**. Earth rotates in the foreground, bathed in stark **Cherenkov-blue and pure-white telemetry light**, while virtual Sentinel-1 (SAR) and Sentinel-2 (Optical) orbital paths sweep across the globe. This visual language drives the entire color palette, glassmorphic HUD panels, split-screen geospatial viewports, and auditable agentic execution traces, while **Gothic & Technical Monospace typography** lends monumental authority to every natural-language query and spatial insight.

---

## 2. Core Visual Scene & Background Architecture

### 2.1 The "Orbital Reconnaissance & Singularity" WebGL Canvas
* **Layer 0 (Deep Space Void):** Pitch-black cosmic void (`#020409`) with a high-precision celestial coordinate grid (`RA/DEC` and `Lat/Lon` graticules) that subtly shifts with cursor parallax.
* **Layer 1 (The Singularity / Accretion Halo):** Positioned in the upper-right backdrop—a dark event horizon ringed by a relativistic accretion halo emitting stark white (`#FFFFFF`) and electric Cherenkov cyan (`#00F0FF` to `#0051FF`). Its pulse rate accelerates when the Agentic Controller is actively routing queries or executing specialist models.
* **Layer 2 (Earth & Swath Telemetry):** Positioned in the lower-left foreground. The globe displays glowing cyan/white vector coastlines and active satellite swath footprints. When a user uploads a GeoTIFF with valid CRS metadata, the 3D globe smoothly rotates and zooms to lock onto the exact bounding box (`BBox`) of the uploaded imagery.
* **Layer 3 (Gravitational & Radar Shader):** A subtle post-processing shader that sends a synthetic radar scanline (`SAR C-Band Pulse`) across the viewport whenever a cross-modal or change-detection workflow is triggered.

---

## 3. Color Palette & Atmospheric Theme

The palette is strictly anchored to a **Stark White and Cherenkov Blue** spectrum against deep void blacks, reserving secondary semantic accents exclusively for spatial grounding overlays and change-detection masks.

### 3.1 Backgrounds & HUD Surfaces
* **Event Horizon (Primary BG):** `#020409` (Deepest Void Black-Blue)
* **Orbital Shadow (Secondary BG / Viewport Well):** `#070D19` (Midnight Telemetry Blue)
* **HUD Panel Glass Surface:** `rgba(10, 22, 47, 0.48)` with `backdrop-filter: blur(20px) saturate(160%)`
* **Interactive Card / Trace Node Surface:** `rgba(18, 38, 76, 0.58)` with subtle inner glow `inset 0 0 15px rgba(0, 240, 255, 0.12)`

### 3.2 Text & Typography Colors
* **Starlight White (Primary Text, Headings & VQA Answers):** `#FFFFFF`
* **Ice Telemetry (Secondary Text, Captions & Scene Descriptions):** `#D6E8FF`
* **Cryo Blue (Muted Metadata, CRS Projections & Band Info):** `#6C98C6`

### 3.3 Accents & Geospatial Overlay Semantics
* **Cherenkov Cyan (Primary Accent / Active Agent Node / Water Mask):** `#00F0FF`
* **Pulsar Blue (Secondary Accent / Optical–SAR Fusion Links):** `#1E6BFF`
* **Grounding Amber (Region Grounding Bounding Boxes & Polygons):** `#FFB800`
* **Change Delta Crimson (Bi-Temporal Positive/Negative Change Mask):** `#FF2E54`
* **Relativistic Glow (Gradients):** `linear-gradient(135deg, #FFFFFF 0%, #00F0FF 45%, #0044FF 100%)`
* **Border / Hairline Divider:** `rgba(0, 240, 255, 0.22)` (`1px solid`, intensifying to `0.65` on active tool execution)

---

## 4. Typography System (Monumental Gothic + Technical Precision)

To balance dramatic visual identity with rigorous remote-sensing interpretability, **SatQuery AI** pairs **Architectural Display Gothic** for brand and module headers with **Neo-Grotesque Sans** for natural-language interactions and **Technical Monospace** for geospatial metadata and auditable execution logs.

### 4.1 Font Families
1. **Display / Hero Headers (Dramatic Gothic):**
   * *Choices:* `'Pirata One'`, `'UnifrakturMaguntia'`, or sharp Neo-Gothic Display.
   * *Usage:* **SatQuery AI** brand wordmark, primary workspace mode headers, and report cover insignias.
2. **UI & Natural-Language Copy (Modern Gothic Sans):**
   * *Choices:* `'Space Grotesk'` or `'Bank Gothic'`.
   * *Usage:* User text queries, VQA responses, scene descriptions, multitemporal change narratives, and interactive controls.
3. **Geospatial Telemetry & Execution Trace (Technical Mono):**
   * *Choices:* `'JetBrains Mono'` or `'Space Mono'`.
   * *Usage:* Coordinate Reference Systems (`EPSG:32632`), Ground Sample Distance (`GSD: 10m`), band combinations (`B04/B03/B02`, `VV/VH`), confidence scores ($C = 0.942$), and JSON execution trace parameters.

### 4.2 Typographic Scale & Treatments
* **Hero Brand (`SatQuery AI`):** `3.75rem` (60px) | Line-height: `1.05` | Letter-spacing: `0.04em` | Text-shadow: `0 0 25px rgba(0, 240, 255, 0.45)`
* **HUD Module Header (H2):** `1.5rem` (24px) | Uppercase | Letter-spacing: `0.12em` | Color: `#FFFFFF`
* **Card / Tool Node Title (H3):** `1.15rem` (18.4px) | Color: `#00F0FF`
* **VLM Response & Query Body:** `1.0rem` (16px) | Line-height: `1.65` | Color: `#D6E8FF`
* **Micro-Telemetry & Sensor Badges:** `0.75rem` (12px) | Uppercase | Letter-spacing: `0.16em` | Color: `#6C98C6`

---

## 5. Layout, HUD Panels & Core Functional Modules

### 5.1 Spatial Workspace Grid
* **Viewport-Locked Command Layout:** 12-column fluid CSS Grid (`100vw x 100vh` application shell with **Lenis** smooth scrolling inside evidence feeds and trace drawers).
* **Left Column (3 Cols):** **Data Ingestion & Sensor Validation Dock** (Single Image, Optical–SAR Pair, Bi-Temporal Pair).
* **Center Stage (6 Cols):** **Multimodal Geospatial Canvas** (Interactive map/image viewer with split-swipe, grounding overlays, and change masks) + **Bottom Natural-Language Query Bar**.
* **Right Column (3 Cols):** **Agentic Execution Trace & Evidence Synthesis Feed** (Observable routing pipeline, confidence gauges, and downloadable report generator).

### 5.2 Command Panels (Glassmorphic HUD Modules)

#### 1. Multimodal Ingestion & Compatibility Gate (`PANEL // 01 // INGEST`)
* **Supported Input Modes:**
  * `[MODE: SINGLE]` — Single Optical/Multispectral or SAR image (`GeoTIFF/TIFF`, or benchmark `PNG/JPEG` for `VRSBench`/`RSVQA`).
  * `[MODE: CROSS-MODAL]` — Co-registered Optical + SAR pair (`Sentinel-2 MSI` + `Sentinel-1 SAR`).
  * `[MODE: BI-TEMPORAL]` — Pre-event ($T_1$) and Post-event ($T_2$) co-registered image pair (`CDVQA`).
* **Automated Compatibility Check:** Displays live validation badges in `JetBrains Mono`:
  * `FORMAT: GEOTIFF [PASS]`
  * `CRS ALIGNMENT: EPSG:4326 [CO-REGISTERED]`
  * `DIMENSIONS: 512x512px | BANDS: 12-CH MSI + 2-CH SAR (VV, VH)`

#### 2. Interactive Dual/Single Geospatial Viewport (`PANEL // 02 // GEO-CANVAS`)
* **Single-Image Grounding View:** Renders text-guided region grounding (`"Highlight the water body referred to in the query"`) with glowing cyan/amber bounding boxes, segmentation contours, and coordinate callouts.
* **Cross-Modal & Bi-Temporal Split-Swipe Curtain:** A tactile vertical/horizontal split slider (`< | >`) allowing users to scrub seamlessly between:
  * **Optical RGB/False-Color** $\leftrightarrow$ **SAR Backscatter (`VV/VH`)**
  * **Time $T_1$ (Pre-change)** $\leftrightarrow$ **Time $T_2$ (Post-change)** with toggleable spatial change probability heatmaps (`#FF2E54` / `#00F0FF`).
* **Band & Layer Switcher:** Floating glassmorphic pill bar to toggle True Color, False Color NIR, SAR Amplitude, Grounding Boxes, and Change Masks.

#### 3. Natural-Language Query & Prompt Deck (`PANEL // 03 // QUERY-DECK`)
* Sleek bottom-docked command input with quick-select representative query chips:
  * *"Describe the land-cover and major objects visible in this image."*
  * *"Highlight the water body referred to in the query."*
  * *"What changed between these two dates, and where did the change occur?"*
  * *"Use the optical and SAR images together to identify built-up and water-covered regions."*
  * *"Has the built-up area increased, decreased, or remained unchanged?"*

#### 4. Auditable Agentic Execution Trace (`PANEL // 04 // AGENT-TRACE`)
Since the evaluation strictly grades the **observable execution trace** (rather than hidden chain-of-thought), this panel renders a live, deterministic **Directed Acyclic Graph (DAG) & Telemetry Log** for every query:
1. **Task Classification Node:** Displays detected intent (e.g., `TASK: CROSS_MODAL_EXTRACTION` or `TASK: BITEMPORAL_CHANGE_VQA`).
2. **Input Verification Node:** Confirms image count, sensor modalities (`OPTICAL + SAR`), and spatial co-registration status.
3. **Model & Tool Registry Selector:** Highlights the activated specialist model(s) from the registry:
   * `BigEarthNet-Adapted VLM Backbone` (Domain-adapted vision-language encoder)
   * `RS-VQA / Captioning Specialist` (Evaluated on `RSVQA` / `VRSBench`)
   * `RS-Grounding Specialist` (Text-guided bounding box / segmentation generator)
   * `Bi-Temporal Change-VQA & Mask Specialist` (Evaluated on `CDVQA`)
   * `Optical–SAR Cross-Modal Fusion Extractor`
4. **Permitted Parameter Inspector:** Displays exact runtime parameters in a collapsible code block (`threshold: 0.45`, `fusion_mode: "late_cross_attention"`, `crs_lock: true`).

---

## 6. Interactive Evidence Flashcards & Report Export

### 6.1 Evidence-Grounded Result Cards
Outputs are presented as tactile, 3D-tilting glass flashcards in the Right Column:
* **Front of Card (Synthesis & Confidence):**
  * Top badge: `[TASK: CHANGE_VQA // CDVQA-ADAPTED]`
  * Main body: Natural-language answer grounded in visual evidence (e.g., *"Built-up area increased by +18.4% in the northeastern quadrant between $T_1$ and $T_2$."*).
  * Bottom telemetry bar: Confidence meter (`CONFIDENCE: 94.2%`) + `[ HOVER TO HIGHLIGHT REGION ]` + `[ FLIP FOR AUDIT LOG ]`.
* **Back of Card (Auditable Execution Summary):**
  * Full JSON/structured breakdown of the executed model name, checkpoint adaptation source (`BigEarthNet.txt`), input tensor shapes, inference latency (`240ms`), and spatial bounding coordinates.

### 6.2 One-Click Downloadable Intelligence Report
* Clicking `[ EXPORT AUDITABLE REPORT ]` compiles the query, input metadata, visual evidence overlays (grounding/change maps), confidence scores, and the complete agentic execution summary into a clean **PDF / GeoJSON / Markdown Report Bundle**.

---

## 7. Award-Winning Micro-Interactions & Details

1. **Spatial Cross-Linking (Hover Beams):** Hovering over an entity or coordinate in the textual VQA answer (`"built-up region in the NE sector"`) draws a glowing cyan SVG bezier beam directly to the corresponding bounding box or polygon on the Geospatial Viewport.
2. **Radar Sweep Loading State:** While specialist models execute, a Cherenkov-cyan synthetic aperture radar scanline sweeps across the uploaded satellite tile, accompanied by a 300ms glyph-decoding scramble (`01X#ΔΣ`) as the execution trace populates.
3. **Dual-Cursor Reticle:** Inside the Geospatial Viewport, the cursor transforms into a precision orbital targeting reticle displaying live pixel coordinates `(X, Y)` and geographic `(Lat, Lon)` when GeoTIFF tags are present.

---

## 8. Component CSS Blueprint (Reference Tokens)

```css
:root {
  /* Atmospheric Color System */
  --bg-void: #020409;
  --bg-viewport: #070D19;
  --bg-hud-panel: rgba(8, 20, 44, 0.52);
  --bg-evidence-card: rgba(14, 34, 70, 0.62);
  
  --text-white: #FFFFFF;
  --text-ice: #D6E8FF;
  --text-muted: #6C98C6;
  
  /* Semantic & Telemetry Accents */
  --accent-cyan: #00F0FF;
  --accent-blue: #1E6BFF;
  --grounding-amber: #FFB800;
  --change-crimson: #FF2E54;
  --border-glow: rgba(0, 240, 255, 0.25);

  /* Typography */
  --font-gothic-display: 'Pirata One', 'UnifrakturMaguntia', serif;
  --font-gothic-ui: 'Space Grotesk', 'Bank Gothic', sans-serif;
  --font-telemetry: 'JetBrains Mono', monospace;

  /* Motion */
  --ease-orbital: cubic-bezier(0.16, 1, 0.3, 1);
}

.satquery-hud-panel {
  background: var(--bg-hud-panel);
  backdrop-filter: blur(20px) saturate(160%);
  -webkit-backdrop-filter: blur(20px) saturate(160%);
  border: 1px solid var(--border-glow);
  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.75),
              inset 0 0 20px rgba(0, 240, 255, 0.08);
  transition: transform 0.45s var(--ease-orbital), border-color 0.3s ease;
}

.satquery-evidence-card {
  background: var(--bg-evidence-card);
  border: 1px solid rgba(255, 255, 255, 0.18);
  transform-style: preserve-3d;
  transition: transform 0.6s var(--ease-orbital), box-shadow 0.4s ease;
}

.satquery-evidence-card:hover {
  border-color: var(--accent-cyan);
  box-shadow: 0 0 30px rgba(0, 240, 255, 0.28),
              inset 0 0 15px rgba(255, 255, 255, 0.12);
}
```

---

## 9. Recommended Tech Stack for Implementation

* **Frontend Application:** Next.js (React) or Vite + React + TypeScript
* **Geospatial & Image Canvas:** Deck.gl / MapLibre GL JS + `geotiff.js` (client-side GeoTIFF parsing, CRS reading, and multispectral/SAR band rendering)
* **3D Orbital Backdrop:** Three.js / React Three Fiber (`@react-three/fiber`, `@react-three/postprocessing` for accretion bloom and radar shaders)
* **Agentic Backend & Model Server:** FastAPI (Python) + PyTorch + Rasterio/GDAL (for GeoTIFF compatibility validation and specialist VLM/Grounding/Change-VQA inference)
* **Motion & Styling:** Tailwind CSS + Framer Motion + GSAP + Lenis Smooth Scroll