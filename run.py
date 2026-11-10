"""SatQuery AI - one-command launcher.

    python run.py                 start the app and open it in your browser
    python run.py --setup-models  download the training data and train the three AI models (~45 min, CPU is fine)
    python run.py --setup-highres optional: models for sharp aerial photos (VRSBench, ~12 GB download)
    python run.py --test          run the automated tests
    python run.py --port 8080     use a different port
"""
from __future__ import annotations

import argparse
import importlib.util
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NEEDED = ["numpy", "rasterio", "fastapi", "uvicorn", "PIL", "cv2", "matplotlib", "reportlab"]
MODELS = ["checkpoints/scene_resnet18.pt", "checkpoints/rsvqa_lr.pt", "checkpoints/cdvqa.pt"]
EUROSAT = "https://huggingface.co/datasets/torchgeo/eurosat/resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip"
EUROSAT_SPLITS = "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/eurosat-{}.txt"
EUROSAT100 = "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSAT100.zip"
RSVQA = "https://zenodo.org/api/records/6344334/files/{}/content"
RSVQA_FILES = [f"LR_split_{s}_{k}.json" for s in ("train", "val", "test") for k in ("questions", "answers", "images")]


def say(msg: str) -> None:
    print(f"  {msg}", flush=True)


def check_python_and_packages() -> bool:
    if sys.version_info < (3, 10):
        say(f"SatQuery needs Python 3.10 or newer (you have {sys.version.split()[0]}).")
        return False
    missing = [m for m in NEEDED if importlib.util.find_spec(m) is None]
    if missing:
        say("Some packages are missing: " + ", ".join(missing))
        say("Install them with:   pip install -r requirements.txt")
        return False
    return True


def free_port(preferred: int) -> int:
    for port in [preferred] + list(range(preferred + 1, preferred + 20)):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    say(f"downloading {dest.name} ...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def setup_models() -> int:
    """Download EuroSAT, RSVQA-LR and a CDVQA subset, and train the three models. Safe to re-run: finished steps are skipped."""
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torchvision") is None:
        say("Training needs PyTorch. Install it with:   pip install torch torchvision")
        return 1
    eu, rv = ROOT / "data/eurosat", ROOT / "data/rsvqa_lr"
    print("\n[1/6] EuroSAT (Sentinel-2 land-use images, ~100 MB)")
    download(EUROSAT, eu / "EuroSAT.zip")
    download(EUROSAT100, eu / "EuroSAT100.zip")
    for s in ("train", "val", "test"):
        download(EUROSAT_SPLITS.format(s), eu / f"eurosat-{s}.txt")
    if not (eu / "rgb").exists():
        zipfile.ZipFile(eu / "EuroSAT.zip").extractall(eu / "rgb")
        zipfile.ZipFile(eu / "EuroSAT100.zip").extractall(eu / "ms100")
    print("\n[2/6] RSVQA-LR (satellite question-answering benchmark, ~130 MB)")
    for f in RSVQA_FILES + ["Images_LR.zip"]:
        download(RSVQA.format(f), rv / f)
    if not (rv / "Images_LR").exists():
        zipfile.ZipFile(rv / "Images_LR.zip").extractall(rv)
    py = [sys.executable]
    if not (ROOT / MODELS[0]).exists():
        print("\n[3/6] Training the land-use classifier (about 15-20 minutes on a laptop CPU)")
        subprocess.run(py + ["training/train_scene_classifier.py", "--data", str(eu), "--out", MODELS[0]], cwd=ROOT, check=True)
    else:
        say("[3/6] land-use classifier already trained - skipping")
    if not (ROOT / MODELS[1]).exists():
        print("\n[4/6] Training the question-answering model (about 3 minutes)")
        subprocess.run(py + ["training/train_rsvqa_lr.py", "--root", str(rv), "--out", MODELS[1]], cwd=ROOT, check=True)
    else:
        say("[4/6] question-answering model already trained - skipping")
    if not (ROOT / MODELS[2]).exists():
        print("\n[5/6] CDVQA change questions (streams ~210 shards, keeps ~600 image pairs; 15-30 minutes)")
        for split, n in (("test", 60), ("train", 150)):
            if not (ROOT / f"eval/data/cdvqa_{split}.jsonl").exists():
                subprocess.run(py + ["eval/fetch_cdvqa.py", "--split", split, "--shards", str(n)], cwd=ROOT, check=True)
        print("\n[6/6] Training the change-question model (about 5 minutes)")
        subprocess.run(py + ["training/train_cdvqa.py", "--out", MODELS[2]], cwd=ROOT, check=True)
    else:
        say("[5-6/6] change-question model already trained - skipping")
    print("\nDone. Start the app with:   python run.py")
    print("Optional, for sharp aerial photos (objects such as planes, ships, vehicles): python run.py --setup-highres")
    return 0


def setup_highres() -> int:
    """VRSBench: high-resolution question answering, descriptions and approximate object boxes (~12 GB download)."""
    if importlib.util.find_spec("torch") is None:
        say("Training needs PyTorch. Install it with:   pip install torch torchvision")
        return 1
    vb = ROOT / "data/vrsbench"
    print("\n[1/3] VRSBench (high-resolution aerial images, about 12 GB; resumes if interrupted)")
    base = "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/"
    for f in ("VRSBench_train.json", "VRSBench_EVAL_vqa.json", "VRSBench_EVAL_referring.json",
              "VRSBench_EVAL_Cap.json", "Images_val.zip", "Images_train.zip"):
        download(base + f, vb / f)
    py = [sys.executable]
    if not (ROOT / "checkpoints/vrsbench_vqa.pt").exists():
        print("\n[2/3] Training the high-resolution question model (about 25 minutes on a laptop CPU)")
        subprocess.run(py + ["training/train_vrsbench_vqa.py", "--root", str(vb), "--epochs", "8"], cwd=ROOT, check=True)
    if not (ROOT / "checkpoints/vrsbench_ground.pt").exists():
        print("\n[3/3] Training the object locator (about 3 minutes)")
        subprocess.run(py + ["training/train_vrsbench_grounding.py", "--root", str(vb), "--epochs", "12"], cwd=ROOT, check=True)
    print("\nDone. Start the app with:   python run.py")
    return 0


def start(port: int, open_browser: bool) -> int:
    print("\nSatQuery AI")
    if not check_python_and_packages():
        return 1
    if not (ROOT / "satquery/web/dist/index.html").exists():
        say("The web interface is not built. Build it with:   cd frontend && npm install && npm run build")
        return 1
    missing = [m for m in MODELS if not (ROOT / m).exists()]
    if missing:
        say("Note: the trained AI models are not installed yet, so answers come from the measurement tools only.")
        say("      To add them (one time, ~45 min):   python run.py --setup-models")
    sys.path.insert(0, str(ROOT))
    from satquery.demo.make import load_manifest
    load_manifest()  # creates the demo images on first run
    port = free_port(port)
    url = f"http://localhost:{port}"
    say(f"Starting at {url}   (press Ctrl+C to stop)")
    if open_browser:
        threading.Thread(target=lambda: (time.sleep(2.5), webbrowser.open(url)), daemon=True).start()
    import uvicorn
    uvicorn.run("satquery.api.app:app", host="127.0.0.1", port=port, log_level="warning")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="SatQuery AI launcher")
    ap.add_argument("--setup-models", action="store_true", help="download data and train the AI models")
    ap.add_argument("--setup-highres", action="store_true", help="optional: high-resolution aerial models (VRSBench, ~12 GB)")
    ap.add_argument("--test", action="store_true", help="run the automated tests")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    a = ap.parse_args()
    if a.setup_models:
        return setup_models()
    if a.setup_highres:
        return setup_highres()
    if a.test:
        return subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT).returncode
    return start(a.port, not a.no_browser)


if __name__ == "__main__":
    sys.exit(main())
