import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KERAS_BACKEND"] = "tensorflow"   # 🔥 IMPORTANT FIX

import cv2
import glob
import numpy as np
import tensorflow as tf


# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
#MODEL_PATH = "/home/frauas/segmentation219_AIS/checkpoints_final/best_model.keras"
#IMAGE_FOLDER = "/home/frauas/segmentation219_AIS/Semantic Segmentation 2026/Output Images after segmentation"
#OUTPUT_FOLDER = "/home/frauas/segmentation219_AIS/segmentation_results"


MODEL_PATH = "checkpoints/best_model.keras"
IMAGE_FOLDER = "input_images"
OUTPUT_FOLDER = "output_images"

IMG_SIZE = 128
NUM_CLASSES = 10


CLASS_COLORS = np.array([
    [0,   0,   0  ],
    [0,   255, 0  ],
    [255, 0,   0  ],
    [0,   0,   255],
    [255, 255, 0  ],
    [255, 0,   255],
    [128, 128, 128],
    [0,   128, 255],
    [255, 128, 0  ],
    [128, 0,   255],
], dtype=np.uint8)


# ─────────────────────────────────────────────
# LOAD MODEL (Safe Version)
# ─────────────────────────────────────────────
print("Loading model...")

try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False
    )
except Exception as e:
    print("Standard load failed. Trying legacy loader...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

print("Model loaded successfully.")


# ─────────────────────────────────────────────
# CREATE OUTPUT FOLDER
# ─────────────────────────────────────────────
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ─────────────────────────────────────────────
# READ IMAGES (Recursive Support)
# ─────────────────────────────────────────────
image_paths = []
for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
    image_paths.extend(
        glob.glob(os.path.join(IMAGE_FOLDER, "**", ext), recursive=True)
    )

if not image_paths:
    print("No images found.")
    exit()

print(f"Found {len(image_paths)} images.")


# ─────────────────────────────────────────────
# PROCESS EACH IMAGE
# ─────────────────────────────────────────────
for path in image_paths:

    print(f"Processing: {path}")

    frame = cv2.imread(path)
    if frame is None:
        continue

    original = frame.copy()

    # Resize + normalize
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    rgb = rgb.astype(np.float32) / 255.0

    # Fake XYZ channels
    xyz = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

    inp = np.expand_dims(np.concatenate([rgb, xyz], axis=-1), axis=0)

    pred = model(inp, training=False)
    mask = tf.argmax(pred[0], axis=-1).numpy().astype(np.uint8)

    colour = CLASS_COLORS[mask]
    colour = cv2.resize(
        colour,
        (original.shape[1], original.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    overlay = cv2.addWeighted(original, 0.6, colour, 0.4, 0)

    # Show result
    cv2.imshow("Segmentation Output", overlay)
    cv2.waitKey(300)

    # Save result
    filename = os.path.basename(path)
    save_path = os.path.join(OUTPUT_FOLDER, filename)
    cv2.imwrite(save_path, overlay)

print("Segmentation complete.")
cv2.destroyAllWindows()
