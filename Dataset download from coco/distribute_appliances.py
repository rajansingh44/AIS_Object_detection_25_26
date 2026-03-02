import os
import shutil

# =============================
# CONFIG
# =============================

SOURCE_DIR = "/home/frauas/segmentation219_AIS/data/frauas_10classes/coco_appliances/train2017"

TARGET_ROOT = "/home/frauas/segmentation219_AIS/data/frauas_10classes"

NUM_SCENES = 5   # 0,1,2,3,4

# =============================
# VERIFY SOURCE
# =============================

if not os.path.exists(SOURCE_DIR):
    raise FileNotFoundError(f"Source folder not found: {SOURCE_DIR}")

images = sorted([
    f for f in os.listdir(SOURCE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

print("Total appliance images found:", len(images))

if len(images) == 0:
    print("⚠️ No images found. Check SOURCE_DIR path.")
    exit()

# =============================
# DISTRIBUTE
# =============================

for idx, img_name in enumerate(images):

    scene_id = idx % NUM_SCENES

    scene_img_dir = os.path.join(TARGET_ROOT, str(scene_id), "images")
    os.makedirs(scene_img_dir, exist_ok=True)

    src_path = os.path.join(SOURCE_DIR, img_name)
    dst_path = os.path.join(scene_img_dir, img_name)

    shutil.copy2(src_path, dst_path)

    print(f"[OK] Scene {scene_id}: {img_name}")

print("\n✅ Appliance images distributed into scenes 0-4")
