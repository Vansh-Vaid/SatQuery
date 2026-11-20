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
| Q7 | New tasks | ~~Flood and burned-area specialists~~ replaced (Q10): no event gets its own model |
| Q8 | Imagery families | Sentinel-1/2 GeoTIFF (have), **Landsat 8/9** GeoTIFF (new), true-colour photos and screenshots (aligned on landmarks); not oblique drone views, hyperspectral or LiDAR |
| Q10 | General design | **One trained land-cover model** (9 Dynamic World classes); every event (flood, forest loss, fire scars, urban growth, drought, cropland, snow loss) is a class change defined in `satquery/tools/events.py` |
| Q11 | Labels | Dynamic World expert labels (PANGAEA, CC-BY) + matching Sentinel-2 L2A from Planetary Computer |
| Q12 | Sen1Floods11 | Test only (water IoU on flood scenes, Bolivia held out), not a separate tool |
| Q13 | Versions | Multispectral and true colour; radar keeps a water-only model |
| Q14 | Burned areas | Dropped as a workstream: fire scars are vegetation -> bare |
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
| C. General land-cover change model | **Code done**: `seg_net` (9 classes), `events.py`, `change_mapper` (replaces the flood tool; used for every change question once trained; gate: beats the rule-based land cover on Dynamic World's held-out experts), `fetch_dynamic_world.py`, `train_landcover.py` (smoke-tested), radar water model. **Waiting on:** `kaggle/satquery_landcover_train.ipynb` |
| E. Free-form VLM | **Gate passed** (Kaggle, 19 Nov 2026): SmolVLM-256M + LoRA (rank 16, 1,374 steps on 12,000 VRSBench rows) scores 53.7% on 300 held-out VRSBench open questions vs 23.7% untrained; captions BLEU-4 0.110 vs 0.043, ROUGE-L 0.323 vs 0.202; 3.2 s per answer on CPU. Adapter: `checkpoints/smolvlm-rs-lora`; scores: `eval/results/smolvlm_{base,lora}.json` |
| D. Burned areas | Dropped (Q14) |
| F. India test set | **Code done**: nine Indian sites in `eval/real_pair_test.py --region india` (with trained-model scoring), per-country flood IoU; runs inside the land-cover notebook |
| G. Docs and merge | After the Kaggle verdicts |

Dataset paths for Sen1Floods11 were checked against the public bucket listing (splits include a Bolivia-only
held-out file, used as the "never seen" test).
