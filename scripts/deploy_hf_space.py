"""Deploy SatQuery AI to a Hugging Face Space (free tier: 2 CPUs, 16 GB RAM, public URL).

One time:
    pip install huggingface_hub
    hf auth login            # paste a token with "write" access from huggingface.co/settings/tokens

Then, from the SatQuery folder (re-run it to update the live app):
    python scripts/deploy_hf_space.py                       # creates <your-name>/satquery
    python scripts/deploy_hf_space.py --space me/my-app --private

It uploads only what the app needs to run (code, built web UI, demo images and the trained models in
checkpoints/) in a single commit; Hugging Face then builds the Dockerfile and starts the app.
If ANTHROPIC_API_KEY is set in your shell, --with-claude stores it as a private Space secret.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["Dockerfile", ".dockerignore", "requirements.txt", "run.py", "satquery", "demo_data", "checkpoints"]
SKIP_PARTS = {"__pycache__", ".pytest_cache"}
SPACE_README = """---
title: SatQuery AI
emoji: 🛰️
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Ask plain-English questions about satellite images
---

# SatQuery AI

Ask questions about satellite images in plain English: what is in a scene, what changed between two dates,
and what optical and radar images show together. Every answer comes with its evidence and a downloadable
report.

Source code, documentation and measured accuracy: {source}
"""


def files_to_upload() -> list[Path]:
    out = []
    for name in INCLUDE:
        p = ROOT / name
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out += [f for f in p.rglob("*") if f.is_file() and not SKIP_PARTS & set(f.parts)
                    and f.suffix != ".pyc" and not (name == "checkpoints" and f.suffix not in (".pt", ".json"))]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", help="owner/name (default: <your-username>/satquery)")
    ap.add_argument("--private", action="store_true", help="only you can open the app")
    ap.add_argument("--with-claude", action="store_true", help="store $ANTHROPIC_API_KEY as a Space secret")
    ap.add_argument("--source", default="https://github.com/Vansh-Vaid/SatQuery")
    ap.add_argument("--dry-run", action="store_true", help="list what would be uploaded, upload nothing")
    args = ap.parse_args()

    if not (ROOT / "satquery/web/dist/index.html").exists():
        print("The web interface is not built. Run:   cd frontend && npm install && npm run build")
        return 1
    models = sorted(p.name for p in (ROOT / "checkpoints").glob("*.pt"))
    if not models:
        print("Note: no trained models in checkpoints/. The app will run on measurement tools only.")
        print("      Train them first with:   python run.py --setup-models")
    files = files_to_upload()
    size = sum(f.stat().st_size for f in files) / 1e6
    print(f"{len(files) + 1} files, {size:.0f} MB; trained models: {', '.join(models) or 'none'}")
    if args.dry_run:
        for f in files:
            print("  ", f.relative_to(ROOT).as_posix())
        return 0

    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError:
        print("Install the Hugging Face client first:   pip install huggingface_hub")
        return 1
    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:
        print("Not logged in to Hugging Face. Run:   hf auth login   (token with write access)")
        return 1
    repo = args.space or f"{user}/satquery"
    api.create_repo(repo, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)

    ops = [CommitOperationAdd(path_in_repo=f.relative_to(ROOT).as_posix(), path_or_fileobj=str(f)) for f in files]
    ops.append(CommitOperationAdd(path_in_repo="README.md",
                                  path_or_fileobj=SPACE_README.format(source=args.source).encode("utf-8")))
    print(f"uploading to {repo} ...")
    api.create_commit(repo, operations=ops, repo_type="space", commit_message="Deploy SatQuery AI")

    if args.with_claude:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if key:
            api.add_space_secret(repo, "ANTHROPIC_API_KEY", key)
            print("stored ANTHROPIC_API_KEY as a private Space secret (Claude planner on)")
        else:
            print("--with-claude: ANTHROPIC_API_KEY is not set in this shell; skipped")

    owner, name = repo.split("/")
    print("\nDone. Hugging Face is now building the app (about 5-10 minutes the first time).")
    print(f"  Build logs and settings: https://huggingface.co/spaces/{repo}")
    print(f"  Your app:                https://{owner.lower()}-{name.lower().replace('_', '-')}.hf.space")
    return 0


if __name__ == "__main__":
    sys.exit(main())
