"""Manual smoke test against a real vision provider (NOT part of the pytest suite).

Usage:  python test_outcrop_vision.py path/to/outcrop.jpg [height_m]
Needs ANTHROPIC_API_KEY or VISION_API_KEY + VISION_BASE_URL in .env.
"""
import os
import shutil
import sys

from config.settings import SEISMIC_UPLOAD_DIR
from core.vision_client import build_vision_client
from workflows.recipes.outcrop_to_seismic import outcrop_to_seismic


def main():
    if len(sys.argv) < 2:
        print(__doc__); return 1
    try:
        client = build_vision_client()
    except RuntimeError as exc:
        print(f"skipped: {exc}"); return 0
    os.makedirs(SEISMIC_UPLOAD_DIR, exist_ok=True)
    staged = os.path.join(SEISMIC_UPLOAD_DIR, os.path.basename(sys.argv[1]))
    shutil.copyfile(sys.argv[1], staged)
    height = float(sys.argv[2]) if len(sys.argv) > 2 else None
    res = outcrop_to_seismic(staged, height_m=height, vision_client=client, display="both")
    print(res["interpretation"]["summary"])
    for r in res["regions"]:
        print(f"  #{r['id']} {r['label']:<20} {r['lithology']:<16} route={r['route']:<10} "
              f"vp={r['vp']} cells={r['n_cells']}")
    print("scale:", res["scale"])
    print("section:", res["grid_shape"], "max|amp| =", round(res["max_abs_amplitude"], 4))
    print("plots:", res["image_path"], res["extra_image_paths"][0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
