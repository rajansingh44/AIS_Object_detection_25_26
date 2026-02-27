import os
import requests
from pycocotools.coco import COCO
from tqdm import tqdm

# ========================
# CONFIG
# ========================

ANN_FILE = "/home/frauas/segmentation219_AIS/data/frauas_10classes/coco_appliances/annotations/instances_train2017.json"
OUTPUT_DIR = "/home/frauas/segmentation219_AIS/data/frauas_10classes/coco_appliances/train2017"

APPLIANCE_CLASSES = [
    "microwave"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================
# LOAD COCO
# ========================

print("Loading COCO annotations...")
coco = COCO(ANN_FILE)

cat_ids = coco.getCatIds(catNms=APPLIANCE_CLASSES)
img_ids = coco.getImgIds(catIds=cat_ids)

print("Total appliance images:", len(img_ids))

# ========================
# DOWNLOAD IMAGES
# ========================

for img_id in tqdm(img_ids):

    img_info = coco.loadImgs(img_id)[0]
    url = img_info["coco_url"]
    file_name = img_info["file_name"]

    save_path = os.path.join(OUTPUT_DIR, file_name)

    if os.path.exists(save_path):
        continue

    try:
        response = requests.get(url, timeout=10)
        with open(save_path, "wb") as f:
            f.write(response.content)
    except:
        continue

print("✅ Appliance image download complete.")
