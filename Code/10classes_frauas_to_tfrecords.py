import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

import tensorflow as tf
import numpy as np
import glob
from collections import defaultdict
from PIL import Image
import gc

# =========================================================
# CONFIGURATION
# =========================================================
NUM_CLASSES = 9
IMG_SIZE = 128
BATCH_SIZE = 12
EPOCHS = 50
LR_INITIAL = 8e-4
LR_MIN = 1e-6

dataset_root = "/home/frauas/segmentation219_AIS/data/frauas_10classes"
tfrecord_dir = os.path.join(dataset_root, "tfrecords_9class")
train_record_dir = os.path.join(tfrecord_dir, "train")
val_record_dir = os.path.join(tfrecord_dir, "val")

CLASS_NAMES = [
    "background", "human", "table", "chair", "robot", 
    "backpack", "free", "laptop","bottle","microwave" 
]

CLASS_WEIGHTS = tf.constant([
    0.05, 15.0, 25.0, 22.0, 40.0, 50.0, 15.0, 3.0, 6.0,3.0
], dtype=tf.float32)

img_height, img_width = 480, 640
val_ratio = 0.2

# =========================================================
# PART 1: TFRECORDS GENERATION (CPU ONLY)
# =========================================================
print("="*80)
print("PART 1: GENERATING TFRECORDS (CPU ONLY)")
print("="*80)

# Force CPU for TFRecords generation
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.keras.backend.clear_session()

os.makedirs(train_record_dir, exist_ok=True)
os.makedirs(val_record_dir, exist_ok=True)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def create_example(rgb_path, x_path, y_path, z_path, label_path):
    rgb_bytes = open(rgb_path, "rb").read()
    x_bytes = open(x_path, "rb").read()
    y_bytes = open(y_path, "rb").read()
    z_bytes = open(z_path, "rb").read()
    label_bytes = open(label_path, "rb").read()
    
    label = np.array(Image.open(label_path))
    unique_classes = np.unique(label)
    present_classes = [int(c) for c in unique_classes if c != 7]
    
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "rgb": _bytes_feature(rgb_bytes),
                "x": _bytes_feature(x_bytes),
                "y": _bytes_feature(y_bytes),
                "z": _bytes_feature(z_bytes),
                "label": _bytes_feature(label_bytes),
                "height": _int64_feature([img_height]),
                "width": _int64_feature([img_width]),
                "classes_present": _int64_feature(present_classes),
            }
        )
    ), present_classes

def collect_and_split_files(root):
    all_samples = []
    class_to_samples = defaultdict(list)
    
    print("\n📂 Collecting files...")
    for scene in range(5):
        label_dir = f"{root}/{scene}/label"
        if not os.path.exists(label_dir):
            continue
        
        labels = sorted(glob.glob(label_dir + "/*.png"))
        print(f"   Scene {scene}: {len(labels)} labels")
        
        for lbl in labels:
            name = os.path.splitext(os.path.basename(lbl))[0]
            paths = {
                "rgb": f"{root}/{scene}/rgb/{name}.jpg",
                "x": f"{root}/{scene}/x/{name}.jpg",
                "y": f"{root}/{scene}/y/{name}.jpg",
                "z": f"{root}/{scene}/z/{name}.jpg",
            }
            
            if not all(os.path.exists(p) for p in paths.values()):
                continue
            
            label = np.array(Image.open(lbl))
            unique_classes = [c for c in np.unique(label) if c != 7]
            
            sample_info = {
                'paths': (paths["rgb"], paths["x"], paths["y"], paths["z"], lbl),
                'classes': unique_classes,
            }
            
            all_samples.append(sample_info)
            
            for cls in unique_classes:
                if cls != 0:
                    class_to_samples[cls].append(len(all_samples) - 1)
    
    print(f"\n📊 Total: {len(all_samples)} samples")
    
    if len(all_samples) == 0:
        raise ValueError("No samples found! Check your data directory.")
    
    print("\n🎯 Stratified split...")
    val_indices = set()
    
    rare_classes = [cls for cls in range(10) if cls != 7 and len(class_to_samples[cls]) < len(all_samples) * 0.1]
    
    for cls in rare_classes:
        samples_with_class = class_to_samples[cls]
        if len(samples_with_class) == 0:
            continue
        
        n_val = max(1, int(len(samples_with_class) * val_ratio))
        np.random.seed(42 + cls)
        val_samples_for_class = np.random.choice(samples_with_class, n_val, replace=False)
        val_indices.update(val_samples_for_class)
    
    remaining_indices = set(range(len(all_samples))) - val_indices
    target_val_count = int(len(all_samples) * val_ratio)
    needed = target_val_count - len(val_indices)
    
    if needed > 0:
        np.random.seed(42)
        additional_val = np.random.choice(list(remaining_indices), min(needed, len(remaining_indices)), replace=False)
        val_indices.update(additional_val)
    
    train_indices = set(range(len(all_samples))) - val_indices
    
    print(f"   Train: {len(train_indices)} ({100*len(train_indices)/len(all_samples):.1f}%)")
    print(f"   Val: {len(val_indices)} ({100*len(val_indices)/len(all_samples):.1f}%)")
    
    train = [all_samples[i]['paths'] for i in train_indices]
    val = [all_samples[i]['paths'] for i in val_indices]
    
    return train, val

def write_tfrecord(samples, outfile, split_name):
    print(f"\n📝 Writing {split_name}...")
    
    written = 0
    with tf.io.TFRecordWriter(outfile) as w:
        for i, (rgb, x, y, z, lbl) in enumerate(samples):
            try:
                ex, _ = create_example(rgb, x, y, z, lbl)
                w.write(ex.SerializeToString())
                written += 1
                
                if (i + 1) % 100 == 0:
                    print(f"   {i+1}/{len(samples)}...", end='\r')
            except Exception as e:
                print(f"\n   ⚠️ Skipped {i}: {e}")
    
    print(f"\n   ✅ {written} samples written")

# Generate TFRecords
train_samples, val_samples = collect_and_split_files(dataset_root)

write_tfrecord(
    train_samples,
    os.path.join(train_record_dir, "train_frauas_9classes.tfrecords"),
    "TRAIN"
)

write_tfrecord(
    val_samples,
    os.path.join(val_record_dir, "val_frauas_9classes.tfrecords"),
    "VAL"
)

print("\n✅ TFRecords generation complete!")

# Clear memory and reset for GPU training
del train_samples, val_samples
gc.collect()
tf.keras.backend.clear_session()

# =========================================================
# PART 2: TRAINING WITH GPU
# =========================================================
print("\n" + "="*80)
print("PART 2: TRAINING WITH RGB-D")
print("="*80)

# Re-enable GPU
del os.environ["CUDA_VISIBLE_DEVICES"]

# Mixed precision
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# GPU setup
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ GPU: {gpus}")
else:
    print("⚠️ CPU only")

print("Class weights:", CLASS_WEIGHTS.numpy())

# Data pipeline
feature_desc = {
    "rgb": tf.io.FixedLenFeature([], tf.string),
    "x": tf.io.FixedLenFeature([], tf.string),
    "y": tf.io.FixedLenFeature([], tf.string),
    "z": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.string),
}

def remap_labels(label):
    remapped = tf.where(label == 8, 7, label)
    remapped = tf.where(label == 9, 8, remapped)
    remapped = tf.where(label == 7, 7, remapped)
    return remapped

def parse_example(example):
    ex = tf.io.parse_single_example(example, feature_desc)
    
    rgb = tf.image.decode_jpeg(ex["rgb"], channels=3)
    x = tf.image.decode_jpeg(ex["x"], channels=1)
    y = tf.image.decode_jpeg(ex["y"], channels=1)
    z = tf.image.decode_jpeg(ex["z"], channels=1)
    
    rgb = tf.image.resize(rgb, (IMG_SIZE, IMG_SIZE))
    x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE))
    y = tf.image.resize(y, (IMG_SIZE, IMG_SIZE))
    z = tf.image.resize(z, (IMG_SIZE, IMG_SIZE))
    
    rgb = tf.cast(rgb, tf.float32) / 255.0
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.float32) / 255.0
    z = tf.cast(z, tf.float32) / 255.0
    
    rgbxyz = tf.concat([rgb, x, y, z], axis=-1)
    
    label = tf.image.decode_png(ex["label"], channels=1)
    label = tf.image.resize(label, (IMG_SIZE, IMG_SIZE), method="nearest")
    label = tf.squeeze(label, -1)
    label = tf.cast(label, tf.int32)
    label = remap_labels(label)
    label = tf.clip_by_value(label, 0, NUM_CLASSES - 1)
    
    return rgbxyz, label

def augment(rgbxyz, label):
    if tf.random.uniform(()) > 0.5:
        rgbxyz = tf.image.flip_left_right(rgbxyz)
        label = tf.image.flip_left_right(label[..., tf.newaxis])[..., 0]
    
    if tf.random.uniform(()) > 0.5:
        rgbxyz = tf.image.flip_up_down(rgbxyz)
        label = tf.image.flip_up_down(label[..., tf.newaxis])[..., 0]
    
    rgb = rgbxyz[..., :3]
    xyz = rgbxyz[..., 3:]
    
    rgb = tf.image.random_brightness(rgb, 0.3)
    rgb = tf.image.random_contrast(rgb, 0.6, 1.4)
    rgb = tf.image.random_saturation(rgb, 0.6, 1.4)
    rgb = tf.clip_by_value(rgb, 0.0, 1.0)
    
    rgbxyz = tf.concat([rgb, xyz], axis=-1)
    
    return rgbxyz, label

def load_dataset(split, augment_data=False):
    files = tf.data.Dataset.list_files(
        os.path.join(tfrecord_dir, split, "*.tfrecords"),
        shuffle=(split == "train")
    )
    ds = files.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        num_parallel_calls=tf.data.AUTOTUNE,
        cycle_length=4
    )
    ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    if split == "train":
        ds = ds.shuffle(buffer_size=200)
    
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

print("\n📂 Loading datasets...")
train_ds = load_dataset("train", augment_data=True)
val_ds = load_dataset("val", augment_data=False)
print("✅ Loaded")

# Model
def conv_block(x, filters, dropout_rate=0.0):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x

def unet_rgbd(img_size=IMG_SIZE, num_classes=NUM_CLASSES):
    inputs = tf.keras.Input((img_size, img_size, 6))
    
    c1 = conv_block(inputs, 48)
    p1 = tf.keras.layers.MaxPooling2D()(c1)
    
    c2 = conv_block(p1, 96)
    p2 = tf.keras.layers.MaxPooling2D()(c2)
    
    c3 = conv_block(p2, 192)
    p3 = tf.keras.layers.MaxPooling2D()(c3)
    
    c4 = conv_block(p3, 384, dropout_rate=0.3)
    p4 = tf.keras.layers.MaxPooling2D()(c4)
    
    c5 = conv_block(p4, 384, dropout_rate=0.4)
    
    u6 = tf.keras.layers.Conv2DTranspose(192, 2, strides=2, padding="same")(c5)
    u6 = tf.keras.layers.Concatenate()([u6, c4])
    c6 = conv_block(u6, 384, dropout_rate=0.3)
    
    u7 = tf.keras.layers.Conv2DTranspose(96, 2, strides=2, padding="same")(c6)
    u7 = tf.keras.layers.Concatenate()([u7, c3])
    c7 = conv_block(u7, 192)
    
    u8 = tf.keras.layers.Conv2DTranspose(48, 2, strides=2, padding="same")(c7)
    u8 = tf.keras.layers.Concatenate()([u8, c2])
    c8 = conv_block(u8, 96)
    
    u9 = tf.keras.layers.Conv2DTranspose(24, 2, strides=2, padding="same")(c8)
    u9 = tf.keras.layers.Concatenate()([u9, c1])
    c9 = conv_block(u9, 48)
    
    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation="softmax", dtype='float32')(c9)
    
    return tf.keras.Model(inputs, outputs, name="UNet_RGBD")

model = unet_rgbd()
print("\n📐 Model created")

# Loss functions
def weighted_focal_loss(y_true, y_pred, gamma=4.0):
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    pt = tf.exp(-ce)
    focal = (1 - pt) ** gamma
    
    weights = tf.gather(CLASS_WEIGHTS, y_true)
    return tf.reduce_mean(weights * focal * ce)

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_oh = tf.one_hot(y_true, NUM_CLASSES)
    
    y_true_fg = y_true_oh[..., 1:]
    y_pred_fg = y_pred[..., 1:]
    
    inter = tf.reduce_sum(y_true_fg * y_pred_fg, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true_fg + y_pred_fg, axis=[1, 2, 3])
    dice = (2 * inter + smooth) / (union + smooth)
    
    return 1 - tf.reduce_mean(dice)

def combined_loss(y_true, y_pred):
    return weighted_focal_loss(y_true, y_pred) + 4.0 * dice_loss(y_true, y_pred)

# Evaluation
def compute_per_class_iou(model, dataset, num_classes):
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        preds = np.argmax(preds, axis=-1)
        labels = labels.numpy()
        
        for c in range(num_classes):
            pred_c = (preds == c)
            label_c = (labels == c)
            intersection[c] += np.logical_and(pred_c, label_c).sum()
            union[c] += np.logical_or(pred_c, label_c).sum()
    
    iou = intersection / (union + 1e-7)
    return iou

# Callbacks
class IoUCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, num_classes, class_names, freq=5):
        super().__init__()
        self.val_ds = val_ds
        self.num_classes = num_classes
        self.class_names = class_names
        self.freq = freq
        self.best_mean_iou = 0.0
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.freq == 0:
            print(f"\n{'='*70}")
            print(f"📊 Epoch {epoch + 1} IoU:")
            print(f"{'='*70}")
            
            iou_scores = compute_per_class_iou(self.model, self.val_ds, self.num_classes)
            
            pred_counts = np.zeros(self.num_classes)
            
            for images, labels in self.val_ds:
                preds = self.model.predict(images, verbose=0)
                pred_labels = np.argmax(preds, axis=-1)
                
                for c in range(self.num_classes):
                    pred_counts[c] += np.sum(pred_labels == c)
            
            total_preds = pred_counts.sum()
            
            print(f"\n{'Class':<12} | {'IoU':>6} | {'Pred%':>6} | Status")
            print("-" * 70)
            
            for i in range(self.num_classes):
                pred_pct = 100 * pred_counts[i] / total_preds if total_preds > 0 else 0
                
                if i == 0:
                    target = 0.75
                elif i in [4, 5]:
                    target = 0.50
                else:
                    target = 0.65
                
                status = "✅" if iou_scores[i] >= target else "🔴" if iou_scores[i] == 0 else "🟡"
                iou_bar = "█" * int(iou_scores[i] * 15)
                
                print(f"{self.class_names[i]:<12} | {iou_scores[i]:>6.3f} | {pred_pct:>5.1f}% | {iou_bar:<12} {status}")
            
            mean_iou = iou_scores.mean()
            print("-" * 70)
            print(f"Mean IoU: {mean_iou:.4f}")
            
            if mean_iou > self.best_mean_iou:
                self.best_mean_iou = mean_iou
                print(f"🎯 New best!")
            
            print(f"{'='*70}\n")

def cosine_decay_with_warmup(epoch):
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return LR_INITIAL * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (EPOCHS - warmup_epochs)
        return LR_MIN + (LR_INITIAL - LR_MIN) * 0.5 * (1 + np.cos(np.pi * progress))

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "best_unet_9class.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.LearningRateScheduler(cosine_decay_with_warmup, verbose=0),
    IoUCallback(val_ds, NUM_CLASSES, CLASS_NAMES, freq=5),
    tf.keras.callbacks.CSVLogger("training_9class.csv"),
]

optimizer = tf.keras.optimizers.Adam(LR_INITIAL)

model.compile(
    optimizer=optimizer,
    loss=combined_loss,
    run_eagerly=False
)

print("\n" + "="*70)
print("🚀 TRAINING START")
print("="*70)
print(f"Epochs: {EPOCHS}")
print(f"Batch: {BATCH_SIZE}")
print(f"LR: {LR_INITIAL} → {LR_MIN}")
print("="*70 + "\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Final evaluation
print("\n" + "="*70)
print("🎯 FINAL RESULTS:")
print("="*70)

iou_scores = compute_per_class_iou(model, val_ds, NUM_CLASSES)

pred_counts = np.zeros(NUM_CLASSES)
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    pred_labels = np.argmax(preds, axis=-1)
    for c in range(NUM_CLASSES):
        pred_counts[c] += np.sum(pred_labels == c)

total_preds = pred_counts.sum()

print(f"\n{'Class':<12} | {'IoU':>6} | {'Pred%':>6} | Progress")
print("-" * 70)

for i in range(NUM_CLASSES):
    pred_pct = 100 * pred_counts[i] / total_preds if total_preds > 0 else 0
    iou_bar = "█" * int(iou_scores[i] * 20)
    print(f"{CLASS_NAMES[i]:<12} | {iou_scores[i]:>6.3f} | {pred_pct:>5.1f}% | {iou_bar}")

print("-" * 70)
print(f"Mean IoU: {iou_scores.mean():.4f}")
print("="*70)

print("\n💾 Saved: best_unet_9class.keras")
print("📊 Log: training_9class.csv")