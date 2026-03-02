import os
import json
import numpy as np
from PIL import Image
from labelme.utils import shapes_to_label

# =========================
# CONFIG
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE_DIR, "data/frauas_10classes")

NUM_SCENES = 5

# Background = 0 automatically
LABEL_MAP = {
    "human": 1,
    "table": 2,
    "chair": 3,
    "robot": 4,
    "backpack": 5,
    "free": 6,
    "laptop": 7,
    "bottle": 8,
    "microwave": 10   # microwave mapped to appliance
}

# ---- alias normalization ----
ALIAS = {
    "person": "human",
    "people": "human",
    "man": "human",
    "woman": "human",
    "table ": "table",
    "chair ": "chair",
}

# =========================
# PROCESS
# =========================

total_written = 0
total_skipped = 0
appliance_mask_count = 0

for scene in range(NUM_SCENES):

    json_dir = os.path.join(DATA_ROOT, str(scene), "json")
    img_dir  = os.path.join(DATA_ROOT, str(scene), "images")
    lbl_dir  = os.path.join(DATA_ROOT, str(scene), "label")

    if not os.path.exists(json_dir):
        print(f"⚠️ Skipping Scene {scene} (no json folder)")
        continue

    os.makedirs(lbl_dir, exist_ok=True)

    json_files = sorted(f for f in os.listdir(json_dir) if f.endswith(".json"))
    print(f"\nScene {scene}: {len(json_files)} JSON files")

    scene_written = 0
    scene_skipped = 0

    for jf in json_files:

        jp = os.path.join(json_dir, jf)

        try:
            with open(jp, "r") as f:
                data = json.load(f)
        except Exception:
            scene_skipped += 1
            continue

        # ---- get image size ----
        if "imageHeight" in data and "imageWidth" in data:
            h, w = data["imageHeight"], data["imageWidth"]
        else:
            img_path = os.path.join(img_dir, data.get("imagePath", ""))
            if not os.path.exists(img_path):
                scene_skipped += 1
                continue
            img = Image.open(img_path)
            w, h = img.size

        shapes = data.get("shapes", [])
        clean_shapes = []

        for shape in shapes:

            if shape.get("shape_type") != "polygon":
                continue

            raw_label = shape.get("label", "")
            label = raw_label.strip().lower()
            label = ALIAS.get(label, label)

            if label not in LABEL_MAP:
                continue

            pts = []
            for p in shape.get("points", []):
                try:
                    pts.append([float(p[0]), float(p[1])])
                except:
                    continue

            if len(pts) >= 3:
                shape["label"] = label
                shape["points"] = pts
                clean_shapes.append(shape)

        if not clean_shapes:
            scene_skipped += 1
            continue

        # ---- rasterize mask ----
        try:
            lbl, _ = shapes_to_label(
                img_shape=(h, w, 3),
                shapes=clean_shapes,
                label_name_to_value=LABEL_MAP,
            )
        except:
            scene_skipped += 1
            continue

        if lbl.max() == 0:
            scene_skipped += 1
            continue

        if 10 in np.unique(lbl):
            appliance_mask_count += 1

        out_png = os.path.join(lbl_dir, jf.replace(".json", ".png"))
        Image.fromarray(lbl.astype(np.uint8)).save(out_png)

        scene_written += 1
        total_written += 1

    total_skipped += scene_skipped
    print(f"  ✔ written: {scene_written}")
    print(f"  ✖ skipped: {scene_skipped}")

print("\n=========================")
print(f"TOTAL LABELS WRITTEN: {total_written}")
print(f"TOTAL SKIPPED:        {total_skipped}")
print(f"Appliance masks:      {appliance_mask_count}")
print("🎉 LABEL GENERATION COMPLETE")
