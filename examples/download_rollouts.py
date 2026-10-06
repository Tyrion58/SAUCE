"""Download a versioned rollout setting from Hugging Face and verify SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

REPOSITORY = "Tyrion279/SAUCE-Qwen3-4B-Rollouts"
REVISION = "qwen-rollouts-v1"
BASE_URL = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("debate", "dylan"), default="debate")
    parser.add_argument("--dataset", choices=("math_500", "mmlu_pro", "bbh"),
                        default="math_500")
    parser.add_argument("--all", action="store_true", help="Download all six settings.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "datasets/qwen3_4b")
    args = parser.parse_args()
    with urllib.request.urlopen(BASE_URL + "manifest.json", timeout=60) as response:
        manifest = json.load(response)
    selected = [entry for entry in manifest["files"] if args.all or
                (entry["protocol"] == args.protocol and entry["dataset"] == args.dataset)]
    if not selected:
        raise ValueError("No setting matched the release manifest")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for entry in selected:
        destination = args.output_dir / Path(entry["path"]).name
        if destination.exists() and sha256(destination) == entry["sha256"]:
            print(f"Verified existing {destination}")
            continue
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=args.output_dir, delete=False) as stream:
                temporary = Path(stream.name)
                with urllib.request.urlopen(BASE_URL + entry["path"], timeout=60) as response:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(chunk)
            if (temporary.stat().st_size != entry["compressed_bytes"]
                    or sha256(temporary) != entry["sha256"]):
                raise ValueError(f"Download checksum mismatch: {entry['path']}")
            os.replace(temporary, destination)
            print(f"Downloaded and verified {destination}")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
