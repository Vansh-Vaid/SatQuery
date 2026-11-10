# SatQuery AI

**Ask a question about a satellite image in plain English, and get an answer with a map, a confidence
score and a record of how the answer was reached.**

![SatQuery landing page](docs/images/landing.jpg)

You give SatQuery one or two satellite images and a question such as *"Is there water in this image?"* or
*"Has the built-up area grown since last year?"*. SatQuery works out which analysis tools the question
needs, runs them, checks their results against each other, and shows you:

- **the answer**, in a sentence;
- **where** it found it, highlighted on the image;
- **how sure** it is (high / medium / low confidence);
- **every step it took**, so you can check its work;
- a **report** you can download (PDF, web page or map file).

---

## What you can ask

SatQuery works with three kinds of input:

| You upload | Example questions |
|---|---|
| **One image** | "Is there a road in this image?" · "Is this a rural or an urban area?" · "Describe this scene." · "How much of the area is water?" |
| **An optical image + a radar image of the same place** | "Use both images to find the built-up and water areas." · "What does the radar see under the clouds?" |
| **Two images of the same place, taken at different dates** | "What changed between these dates, and where?" · "Has the built-up area increased, decreased or stayed the same?" |

No images of your own? The app comes with **demo scenes** you can try straight away.

> **Optical** images are normal photos from space, like Google Earth. **Radar** (SAR) images are made by a
> satellite that sends out microwaves; they see through clouds and work at night. More words in the
> [glossary](#glossary).

---

## Get started (about 10 minutes)

You need **Python 3.10 or newer**. Check with `python --version`. If it's missing, install it from
[python.org](https://www.python.org/downloads/); on Windows, tick **"Add Python to PATH"** in the installer.
You don't need a graphics card.

**1. Download the project**

```bash
git clone https://github.com/Vansh-Vaid/SatQuery.git
cd SatQuery
```

No git? On the GitHub page, click **Code → Download ZIP**, unzip it, and open a terminal in that folder.

**2. Install what it needs** (one time, a few minutes)

```bash
python -m venv .venv
```

Then switch the terminal to that environment:

- Windows: `.venv\Scripts\activate`
- macOS / Linux: `source .venv/bin/activate`

and install:

```bash
pip install -r requirements.txt
```

**3. Start it**

```bash
python run.py
```

Your browser opens at **http://localhost:8000**. Click **Open console**, pick a demo scene, and press
**Analyse**. Press `Ctrl + C` in the terminal to stop the app.

### Optional: add the trained AI models (one time, about 45-60 minutes)

The app works straight away using its measurement tools. For much better answers, add the three AI models
trained for satellite images: one recognises land types, one answers questions about a single image, and
one answers questions about change between two images.

```bash
python run.py --setup-models
```

This downloads three public datasets and trains the models on your own computer. A normal laptop is
enough. It keeps about 1 GB on disk, but the change dataset is streamed from a copy that bundles the
images with every question, so expect about 17 GB of download traffic. If it stops part-way, run the
same command again; it picks up where it left off.

For **sharp aerial photos** (objects such as planes, ships and vehicles), there is one more optional step.
It downloads about 12 GB, so it is kept separate:

```bash
python run.py --setup-highres
```

The **large vision-language model** (free-form answers, written captions, optical + radar together) needs a
graphics card. You can train it for free on Kaggle with
[`kaggle/satquery_vlm.ipynb`](kaggle/satquery_vlm.ipynb): upload it to Kaggle, turn on a GPU and the
internet, and click **Save & Run All**. It runs for up to 12 hours on its own and gives you a zip file with
the trained model and its scores. The steps are at the top of the notebook.

---

## Using the app

![SatQuery console](docs/images/console.png)

The console has four panels, numbered in the order you use them:

1. **Inputs** (left): choose *Single image*, *Optical + radar* or *Before & after*. Then pick a demo
   scene or drop in your own image files (GeoTIFF, `.tif`). SatQuery checks your files straight away and
   warns you if something is off, for example two images that don't cover the same place.
2. **Viewer** (middle): your images. Switch between true colour, infrared and radar views, and drag the
   slider to compare two images.
3. **Ask** (middle, below): type a question or click one of the suggestions, then press **Analyse**.
4. **Answer** (right): the answer and its confidence. Hover a highlighted word to see where it is on the
   image. Click **Show audit log** for the technical details, and scroll down for the step-by-step trace.

Below the answer, click **Download report bundle** for everything in one ZIP, or pick a single format
(PDF, Markdown, HTML, GeoJSON map file, or the JSON trace).

---

## How it works

```
 your images + your question
            │
            ▼
   1. Check the inputs ......... right file type? same place? dates in order?
            │
            ▼
   2. Plan ..................... which tools does this question need, in what order?
            │
            ▼
   3. Run the tools ............ AI models + measurement tools (water, vegetation, buildings, change...)
            │
            ▼
   4. Cross-check .............. do the tools agree? agreement raises confidence, disagreement lowers it
            │
            ▼
 answer + map + confidence + full record of every step
```

SatQuery is **agentic**: it decides for itself which tools to use for each question. It only picks from a
fixed list of approved tools, and only within safe settings, so its behaviour stays predictable and every
step is recorded.

If you have an Anthropic API key, Claude does the planning. Without one, built-in rules plan instead; both
are checked by the same safety validator.

---

## How accurate is it?

Everything below was measured on test data the models never saw during training.

| What was tested | Result |
|---|---|
| Recognising the type of land (forest, river, industrial, ...) in 5,399 test images | **95.7%** correct |
| The same, on 100 raw satellite files loaded the way a user upload is | **93%** correct |
| Answering questions about satellite images (RSVQA benchmark, 10,004 questions) | **84.6%** correct |
| – "Is there a ...?" questions | 90.7% |
| – "Are there more X than Y?" questions | 89.2% |
| – "Is this rural or urban?" | 88.0% |
| – "How many ...?" (answered as a range, e.g. *between 11 and 100*) | 72.0% |
| Answering questions about **what changed** between two real aerial images (CDVQA benchmark, 6,000 questions) | **65.1%** correct |
| – "Did X change / increase / decrease?" | 69–79% |
| – "What did X change into?" | 55.0% |
| – "What percentage changed?" (in 10% ranges) | 32.0% |
| Answering questions about **sharp aerial photos** (objects such as planes, ships, vehicles; VRSBench, 37,409 questions) | **54.0%** correct |
| – "Is there a ...?" | 87.5% |
| – "Where is the ...?" (top-left, bottom-right, ...) | 37.1% |
| Drawing a box around a described object in a sharp aerial photo (VRSBench, 16,159 descriptions) | **14.5%** correct |
| Mapping water, vegetation and built-up land on **real optical + radar pairs** (6 cities, about 2 million pixels, checked against the ESA WorldCover map) | **80%** of pixels correct with both images (optical alone 61%, radar alone 74%) |
| – under monsoon clouds (Dhaka) | 78% with both images (optical alone 35%) |

![Optical + radar test on six real cities](docs/images/real_pairs.jpg)

Before the trained models were added, the single-image test scored 41.8%, the change test 18.7% and the
sharp-photo test 23.8%, about as good as guessing. Object boxes were never found at all (0%). For comparison, the research team that created the change benchmark reported 67.7%
with their own model.

## What it can't do yet

Being honest about the limits:

- **Free-form questions** ("what is next to the lake?") are answered weakly. That needs a larger
  vision-language model, which has to be trained on a graphics card (a ready-to-run Kaggle notebook is
  included; see Get started).
- **Captions** are short sentences built from the models' own answers about the image (setting, main
  object, how many, where) plus the measured land cover. They are not written freely.
- **Object boxes** in sharp aerial photos are approximate: right about 1 time in 7, mostly for large
  objects such as sports fields and airplanes, rarely for small vehicles. The app always shows them with
  low confidence.
- **Change questions** are right about two times in three on real image pairs (65%, see above). The
  hardest are "what percentage changed?" and "what is the smallest change?".
- **Optical + radar maps** are right for about 4 pixels in 5 on real image pairs (see above). Bare soil
  is rarely identified correctly, and the optical cloud check sometimes mistakes bright roofs or sand for
  cloud (up to 6% of a clear scene).
- Confidence scores are rule-based estimates, not calibrated probabilities.

---

## Put it online

SatQuery runs as a website anyone can open. It needs about 0.6 GB of memory and no graphics card.

**Live demo:** https://satquery.happysky-a511d77c.koreacentral.azurecontainerapps.io (it switches off when
nobody is using it, so the first visit after a pause takes about a minute).

### How the live demo is hosted: GitHub builds it, Azure runs it

1. **GitHub builds the app** for free on every push to `main`
   ([`.github/workflows/build-image.yml`](.github/workflows/build-image.yml)). The trained models are not
   in git, so the build downloads them from the `models-v1`
   [release](https://github.com/Vansh-Vaid/SatQuery/releases/tag/models-v1). The finished app is
   published as `ghcr.io/vansh-vaid/satquery`.
2. **Azure Container Apps runs it** (an Azure for Students subscription is enough). One time, install the
   Azure CLI and log in:
   ```bash
   winget install -e --id Microsoft.AzureCLI      # then open a new terminal
   az login
   ```
   Then, after each successful GitHub build:
   ```bash
   python scripts/deploy_azure.py
   ```
   It starts (or updates) the app with 1 CPU and 2 GB and prints the link. The app switches off when
   idle, so it uses very little credit. To remove everything: `az group delete --name satquery-rg`.

If you retrain a model, upload the new files to the release (`gh release upload models-v1
checkpoints/*.pt checkpoints/*.json --clobber`), then run the build again from the **Actions** tab.

Azure for Students only allows some regions (the script uses `koreacentral`; check yours with
`az policy assignment list`) and does not allow Azure's own build service, which is why GitHub builds the
app.

### Hugging Face Space (needs a paid PRO account)

Hugging Face now charges for Docker Spaces. With PRO, `python scripts/deploy_hf_space.py` uploads the app
and models and prints a `*.hf.space` link. Options: `--private`, `--space name/other-name`, and
`--with-claude`, which stores your `ANTHROPIC_API_KEY` as a private secret.

### Any server with Docker

```bash
docker compose up -d        # then open http://your-server:8000
```

Train the models first (`python run.py --setup-models`); they are built into the image from `checkpoints/`.

### What the online version does to stay up

- **Fair use:** each visitor can run 20 analyses a minute, and 2 analyses run at once. When the server is
  busy, people see "try again in a minute" instead of waiting forever.
- **Fast landing page:** the demo answers are worked out once when the server starts and shared with every
  visitor.
- **Size limits:** 100 MB per image and 1,000 characters per question.
- **Privacy:** uploaded images are deleted as soon as the answer is ready. Reports stay in memory for the
  last 100 analyses and are never written to disk.

All limits can be changed with environment variables (see [docs/TECHNICAL.md](docs/TECHNICAL.md)).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` is not recognised | Install Python from python.org and tick "Add Python to PATH", then open a new terminal. On macOS/Linux, try `python3`. |
| `pip install` fails on **rasterio** | Upgrade pip first: `python -m pip install --upgrade pip`, then try again. |
| "Some packages are missing" | You may be outside the virtual environment. Run the activate command from step 2 again. |
| Port 8000 is already in use | Nothing to do: SatQuery picks the next free port and prints the address. Or choose one: `python run.py --port 8080`. |
| The browser didn't open | Open the address printed in the terminal (usually http://localhost:8000) yourself. |
| "The trained AI models are not installed yet" | That's a note, not an error. The app works; run `python run.py --setup-models` for better answers. |
| My image is rejected | SatQuery reads GeoTIFF files (`.tif`) with location information. For two images, they must cover the same area. The Inputs panel explains what is wrong. |

To check everything is working, run the automated tests: `python run.py --test`.

---

## Glossary

| Term | Meaning |
|---|---|
| **Optical image** | A picture taken in visible light (and sometimes infrared), like a photo. Clouds block it. |
| **Multispectral** | An optical image with extra colour bands beyond red, green and blue, such as infrared, which shows plant health. |
| **Radar / SAR** | Synthetic Aperture Radar. The satellite sends microwaves and measures the echo. It sees through clouds and works at night. |
| **Sentinel-1 / Sentinel-2** | Free European satellites: Sentinel-1 takes radar images, Sentinel-2 takes optical images. |
| **Bi-temporal** | Two images of the same place taken at different dates, used to find what changed. |
| **GeoTIFF** | An image file that also stores where on Earth it is. |
| **VQA** | Visual Question Answering: answering a question about an image. |
| **Confidence** | How sure SatQuery is about an answer, based on how strong and how consistent the evidence is. |

---

## For developers

- **[Technical reference](docs/TECHNICAL.md):** architecture, the tool list, configuration, training,
  evaluation commands and the full results.
- **[Design decisions](docs/DECISIONS.md):** what was chosen, what the alternatives were, and why.

| Folder | What's inside |
|---|---|
| `satquery/` | The application: input checks, planner, tools, web server |
| `frontend/` | Source code of the web interface (React). A ready-built copy is already included. |
| `training/` | Scripts that train the AI models |
| `eval/` | Scripts that measure accuracy on benchmarks |
| `tests/` | Automated tests |
| `demo_data/` | The demo scenes |

Handy commands:

```bash
python run.py                  # start the app
python run.py --setup-models   # download data and train the AI models
python run.py --test           # run the tests
python run.py --help           # all options
```
