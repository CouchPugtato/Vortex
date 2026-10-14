"""Apply reproducible scenario noise/blur to Blender images once."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from generate import degrade

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", type=Path, required=True)
args = parser.parse_args()
manifest = json.loads((args.dataset / "manifest.json").read_text())
if manifest["backend"] != "blender" or manifest.get("postprocessed"):
    raise ValueError("Requires fresh Blender renders; do not apply effects twice")
rng = np.random.default_rng(manifest["seed"])
images = []
for frame in manifest["frames"]:
    path = args.dataset / frame["image"]
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    noise_rng = np.random.default_rng(frame["case"]["noise_seed"]) if "noise_seed" in frame["case"] else rng
    images.append((path, degrade(image, frame["case"], noise_rng)))
for path, image in images:
    if not cv2.imwrite(str(path), image):
        raise OSError(path)
manifest["postprocessed"] = True
(args.dataset / "manifest.json").write_text(json.dumps(manifest, indent=2))
