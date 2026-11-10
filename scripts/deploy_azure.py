"""Run SatQuery AI on Azure Container Apps (works with an Azure for Students subscription).

The app image is built for free by GitHub Actions (.github/workflows/build-image.yml) and published as
ghcr.io/<owner>/satquery. Azure only runs it, so no Azure build service or registry is needed (Azure for
Students blocks those).

One time:
    winget install -e --id Microsoft.AzureCLI     # then open a new terminal
    az login
    # push to main (or run "Build app image" in the GitHub Actions tab) and make the package public:
    # GitHub > your profile > Packages > satquery > Package settings > Change visibility > Public

Then (re-run it after each new image to update the live app):
    python scripts/deploy_azure.py
    python scripts/deploy_azure.py --location eastasia     # if your subscription blocks the default region

The app gets 1 CPU / 2 GB. With --min-replicas 0 (the default) it switches off when nobody uses it, so it
costs almost nothing while idle; the first visit after a pause takes about a minute.

Delete everything when you no longer need it:
    az group delete --name satquery-rg
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time


def az(*args: str, capture: bool = False, check: bool = True) -> str:
    exe = shutil.which("az") or shutil.which("az.cmd")
    if not exe:
        sys.exit("Azure CLI not found. Install it (winget install -e --id Microsoft.AzureCLI), open a new "
                 "terminal and run 'az login'.")
    res = subprocess.run([exe, *args], text=True, capture_output=capture)
    if check and res.returncode != 0:
        if capture:
            sys.stderr.write(res.stderr)
        sys.exit(f"az {' '.join(args[:2])} failed (exit {res.returncode})")
    return (res.stdout or "").strip() if res.returncode == 0 else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default="ghcr.io/vansh-vaid/satquery:latest")
    ap.add_argument("--name", default="satquery")
    ap.add_argument("--resource-group", default="satquery-rg")
    ap.add_argument("--environment", default="satquery-env")
    ap.add_argument("--location", default="koreacentral")
    ap.add_argument("--cpu", default="1.0")
    ap.add_argument("--memory", default="2.0Gi")
    ap.add_argument("--min-replicas", default="0", help="0 = switch off when idle (cheapest)")
    args = ap.parse_args()

    rg, name = args.resource_group, args.name
    exists = az("containerapp", "show", "-n", name, "-g", rg, "--query", "name", "-o", "tsv",
                capture=True, check=False)
    if exists:
        print(f"Updating {name} to the newest {args.image} ...")
        # a new revision suffix forces Azure to pull the image again even though the tag is the same
        az("containerapp", "update", "-n", name, "-g", rg, "--image", args.image,
           "--revision-suffix", time.strftime("r%m%d%H%M%S"), "--output", "none")
    else:
        if not az("group", "exists", "-n", rg, capture=True) == "true":
            az("group", "create", "-n", rg, "-l", args.location, "--output", "none")
        if not az("containerapp", "env", "show", "-n", args.environment, "-g", rg, "--query", "name", "-o", "tsv",
                  capture=True, check=False):
            print("Creating the Container Apps environment (a few minutes, one time) ...")
            az("containerapp", "env", "create", "-n", args.environment, "-g", rg, "-l", args.location,
               "--output", "none")
        print(f"Creating {name} from {args.image} ...")
        az("containerapp", "create", "-n", name, "-g", rg, "--environment", args.environment,
           "--image", args.image, "--ingress", "external", "--target-port", "7860",
           "--cpu", args.cpu, "--memory", args.memory,
           "--min-replicas", args.min_replicas, "--max-replicas", "1", "--output", "none")
    fqdn = az("containerapp", "show", "-n", name, "-g", rg,
              "--query", "properties.configuration.ingress.fqdn", "-o", "tsv", capture=True)
    print(f"\nLive at: https://{fqdn}\n(The first start takes 1-2 minutes while the models load.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
