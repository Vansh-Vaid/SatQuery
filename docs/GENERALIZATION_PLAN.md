# Generalising SatQuery: plan and decisions

Branch: `generalize`. Budget: 30 hours over one week, before the viva. Nothing reaches `main` or the live app
until it passes its gate (below); the current version stays live as the fallback.

## Decisions (from the planning interview)

| # | Decision | Choice |
|---|---|---|
| Q1 | What "general" means | **Any imagery** (a), **any question** (b), **any task** (c), **any place** (d) |
| Q2 | Timing | Viva next week; work happens now, on a branch |
| Q3 | Cost | Free only: no paid APIs, no GPU hosting; training on the laptop or free Kaggle sessions |
| Q4 | Budget | One week, at most 30 hours |
| Q5 | Safety | `generalize` branch; each feature merges only after it beats the current version on a fixed test |
| Q6 | Free-form questions | A tiny VLM (SmolVLM, 256M/500M) fine-tuned on Kaggle and served on the CPU, with measured confidence |
| Q7 | New tasks | **Floods** (Sen1Floods11) and **burned areas** (CaBuAr) as trained specialists |
| Q8 | Imagery families | Sentinel-1/2 GeoTIFF (have), **Landsat 8/9** GeoTIFF (new), true-colour photos and screenshots (aligned on landmarks); not oblique drone views, hyperspectral or LiDAR |
| Q9 | Proof of "any place" | A held-out **India** test set (Kerala 2018 and Assam 2022 floods, Indian cities) scored against ESA WorldCover and published flood maps |

Decisions taken during execution (owner's veto welcome):
- Landsat bands are mapped onto Sentinel-2 names (B2-B4 -> B02-B04, B5 -> B08, B6/B7 -> B11/B12), so every
  existing optical tool works on Landsat unchanged; Collection-2 Level-2 values are scaled with USGS's factor
  (reflectance = 0.0000275 x DN - 0.2).
- Flood and burned-area specialists share one small segmentation design (a light U-Net on 256 x 256 tiles,
  fast on one CPU core), so a third task later is a dataset and a notebook, not new code.
- SmolVLM-256M is the first candidate (it fits the 2 GB server next to the other models); 500M only if the
  server memory allows it.
- Data downloads and training run in Kaggle notebooks: the development network blocks Python HTTPS.

## Workstreams, hours and gates

| Workstream | Hours | Gate to merge |
|---|---|---|
| A. Landsat 8/9 input (band mapping, scaling, detection) | 3 | Loads stacked and named Landsat GeoTIFFs; land-cover tools agree with the same scene in Sentinel-2 bands; all tests pass |
| B. Screenshot pairs (landmark alignment, cloud and label masking, flood wording) | done | Already measured: alignment residual 1.2 px on the Nepal pair; control test for the river-bed rule |
| C. Flood specialist (Sen1Floods11, Kaggle) | 7 | Flood IoU on Sen1Floods11's held-out test split above the measured-tools baseline; India flood events |
| D. Burned-area specialist (CaBuAr, Kaggle) | 4 | Burned-area IoU on CaBuAr's held-out folds above a dNBR-threshold baseline |
| E. Free-form VLM on CPU (SmolVLM + LoRA, Kaggle) | 8 | Beats the untrained SmolVLM on VRSBench open questions; under 15 s per answer on the server; fits in memory |
| F. India test set (Kaggle notebook, Planetary Computer) | 4 | Runs end to end; reported, whatever the numbers |
| G. Docs, report "future scope", merge | 2 | |

Order: A, then C and E (they need Kaggle runs, so the notebooks go first and train while the integration is
written), then D, F, G.

## Status

| Workstream | State |
|---|---|
| A. Landsat 8/9 | **Done** (`b59147a`): stacked or named Collection-2 files load with Sentinel-2 band roles; land cover matches the Sentinel-2 version of the same scene on > 90% of pixels; 3 tests |
| B. Screenshot pairs | **Done** (on `main`): landmark alignment, cloud/label masking, flood wording, river-bed shape check |
| C. Flood specialist | **Code done** (`5e7fcc4`): `seg_net`, `train_flood.py` (smoke-tested on a fake Sen1Floods11), `flood_mapper` tool, planner/controller routing, gate inside the tool (a variant is used only if it beat the classic method), Bolivia held-out score. **Waiting on:** `kaggle/satquery_flood_train.ipynb` run |
| E. Free-form VLM | **Code done** (`4e883ac`): SmolVLM-ready backend and LoRA targets, VRSBench chat rows with the app's prompts, CPU-timed scorer. **Waiting on:** `kaggle/satquery_smolvlm_train.ipynb` run, then a memory/latency check before serving |
| D. Burned areas | Next: reuses `seg_net`; needs the CaBuAr loader and notebook |
| F. India test set | Next: Kaggle notebook over Planetary Computer (Kerala 2018, Assam 2022, cities) |
| G. Docs and merge | After the Kaggle verdicts |

Dataset paths for Sen1Floods11 were checked against the public bucket listing (splits include a Bolivia-only
held-out file, used as the "never seen" test).
