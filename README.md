# Enhancing ROSWITHA's visual perception through deep learning which is based on 10-Class segmentation model

---

##  Introduction

This project focuses on enhancing the visual perception capabilities of the assistive robot ROSWITHA (RObot System WITH Autonomy) by implementing a deep learning-based 10-class semantic segmentation model.

The system enables real-time scene understanding in complex indoor environments by combining RGB and depth (XYZ) data. The model extends a previous 7-class segmentation framework by introducing three new object classes:

- Laptop  
- Backpack  
- Microwave  

However, the extension introduces severe class imbalance challenges, where rare classes (e.g., backpack: 0.0035%) are significantly underrepresented, leading to training instability and catastrophic forgetting.

---

##  Technologies Used

| Category | Technology |
|--------|----------|
| Language | Python 3 |
| Deep Learning | TensorFlow / Keras |
| Vision | OpenCV |
| Data | NumPy, TFRecords |
| Annotation | LabelMe / CVAT |
| Robotics | ROS2 (Humble) |
| Hardware | NVIDIA RTX 3090 |
| Build System | Colcon |

---

##  Architecture

<img width="1200" height="900" alt="image" src="https://github.com/user-attachments/assets/b5eecdda-c597-448d-a51b-59487f994d1b" />



---

## Flow Diagram

<img width="1200" height="896" alt="Flow Diagram" src="https://github.com/user-attachments/assets/905f6a34-f2f5-4992-ab70-ea7f1e2e0668" />

### Flow Explanation

The pipeline follows a structured multi-stage process from raw data to evaluated predictions. It begins with raw RGB images paired with JSON polygon annotations, which are converted into pixel-wise segmentation masks through label generation — each polygon is filled with its class index to produce dense ground-truth labels. Next, during multi-modal data preparation, depth channels (X, Y, Z) are concatenated with the RGB channels to form a 6-channel input tensor that encodes both appearance and spatial geometry. These paired inputs and masks are serialised into TFRecord format for efficient, bottleneck-free data loading. The dataset is then split into training and validation subsets using stratified sampling to preserve class distribution, with particular attention paid to rare classes. During training, TFRecords are parsed, augmented (flips, brightness/contrast jitter, small rotations), and fed into a U-Net-based encoder-decoder model that outputs a per-pixel probability distribution over all ten classes. A combined weighted focal loss and Dice loss guides optimisation under class imbalance. At inference time, a softmax layer produces normalised class probabilities and an argmax operation selects the most likely class per pixel, generating the final segmentation mask. The model's quality is then assessed through IoU, pixel accuracy, and class-wise analysis, completing the loop from raw annotation to quantitative evaluation.

```
RGB + JSON → Mask Generation → TFRecords → Train/Val Split
         → Augmentation → Model Training (Focal + Dice Loss)
         → Trained .keras Model
         → ROS2 Inference Node → Real-time Segmentation Output
         → Evaluation (IoU, Pixel Accuracy)
```

- **Step 1:**  
  `10classes_frauas_json_to_label.py`  
  → Converts annotation JSON files into pixel-wise segmentation masks.

- **Step 2:**  
  `10classes_frauas_to_tfrecords.py`  
  → Converts images + masks into TFRecord format (optimized for TensorFlow).

- **Step 3:**  
  `train10_fixed.ipynb`  
  → Loads TFRecords and trains the segmentation model.

- **Step 4:**  
  Best model checkpoint is saved for inference/ROS2 deployment.

---
##  What Each File Does : [Code](https://github.com/rajansingh44/AIS_Object_detection_25_26/tree/main/Main%20Code)

### 1️⃣ `10classes_frauas_json_to_label.py`

- **Input:** JSON annotations (FRA-UAS / COCO)  
- **Output:** Pixel-wise segmentation masks  
- **Purpose:** Convert annotations → training-ready labels  

---

### 2️⃣ `10classes_frauas_to_tfrecords.py`

- **Input:** Images + generated masks  
- **Output:** TFRecord dataset  
- **Purpose:** Efficient data pipeline for TensorFlow training  

---

### 3️⃣ `train10_fixed.ipynb`

- **Input:** TFRecords  

- **Performs:**
  - Model loading (custom CNN)  
  - Conservative fine-tuning  
  - Training + validation  

- **Output:**
  - Best model (`.keras`)  
  - Metrics (IoU, Accuracy)  
---

## Methodology

### Data Preparation

The dataset was captured using an Intel RealSense D435 camera mounted on the ROSWITHA platform, producing approximately 50,000 frames at 640×480 resolution. Each frame was manually annotated using the CVAT tool, where object boundaries were marked as polygon coordinates stored in JSON format alongside the corresponding RGB image. During the label generation stage, these polygon annotations were programmatically converted into pixel-wise segmentation masks — each polygon was filled with a unique integer class index, producing a dense ground-truth image where every pixel carries a semantic label. Data cleaning steps including label normalisation and filtering of geometrically invalid or degenerate shapes were applied to ensure annotation consistency before training.

The dataset exhibits extreme class imbalance: background pixels account for 94.15% of all labelled data, while the rarest class (backpack) constitutes only 0.0035%, resulting in an imbalance ratio of approximately 26,900:1. This distribution is far more severe than the 10:1 to 100:1 ratios typically discussed in segmentation literature, and it fundamentally shaped every subsequent design decision in the pipeline.

### Multi-Modal Input
```
Input: H × W × 6 (RGB + XYZ)
```

### Model Output

A key design choice was the incorporation of depth information alongside colour data. For each RGB image, three additional channels representing the X, Y, and Z spatial coordinates were extracted from the depth stream. These six channels were concatenated to form a single input tensor of shape H × W × 6, where H and W are the spatial dimensions (resized to 128×128 for training). The rationale is that indoor environments frequently present colour ambiguity — a white chair against a white table is nearly invisible in RGB but clearly separated by depth, while two objects at the same depth are easily distinguished by colour. Providing both modalities lets the model leverage whichever is more discriminative for a given context. Since naive concatenation leaves the network to discover modality weighting on its own, the architecture incorporates a dedicated fusion module that learns adaptive, channel-wise attention weights.

### Architecture: FRA-UAS-SS Model

The FRA-UAS-SS (Frankfurt University of Applied Sciences Semantic Segmentation) model is built on an encoder-decoder backbone derived from FC-HarDNet. Unlike standard DenseNet, which connects every layer to every preceding layer within a dense block, HarDNet uses a "harmonic" connectivity pattern where each layer connects only to a logarithmically spaced subset of prior layers. This reduces memory bandwidth consumption by roughly 40% compared to DenseNet while achieving comparable accuracy — an important consideration given that the model must eventually run on embedded GPUs aboard the robot.

The encoder consists of four HarDNet blocks with progressively increasing channel depths (64, 128, 256, 512), separated by max-pooling layers that halve the spatial resolution at each stage. The decoder mirrors this structure, using transposed convolutions for upsampling. Following the U-Net principle, skip connections bridge each encoder level to its corresponding decoder level at four resolution scales (128×128, 64×64, 32×32, 16×16), preserving fine-grained spatial details that would otherwise be lost during downsampling.

For multi-modal fusion, the model adopts the MMAF-Net (Multi-Modal Adaptive Fusion) approach. Rather than simply concatenating RGB and XYZ channels at the input, the fusion module processes each modality through separate convolutional streams (three layers each, with 64, 128, and 256 channels). A lightweight attention network then computes channel-wise fusion weights by applying global average pooling to the concatenated features, followed by two fully connected layers and a sigmoid activation. The resulting weight vector α ∈ [0,1]^C determines, for each feature channel, how much to draw from the RGB stream versus the depth stream. This allows the model to dynamically adjust its reliance on colour versus geometry depending on the scene content.

To further refine feature representations, Squeeze-and-Excitation (SE) blocks are placed after each dense block in both the encoder and decoder. Each SE block first "squeezes" spatial information into a per-channel descriptor via global average pooling, then "excites" it through a two-layer bottleneck network (with a reduction ratio of 16) that learns inter-channel dependencies. The recalibrated features allow the network to amplify channels relevant to rare object classes and suppress noisy ones — providing learned class-aware feature weighting at every stage.


### Loss Function Design

The loss function combines two complementary objectives. The focal loss component (with focusing parameter γ = 2) addresses class imbalance by down-weighting the contribution of easily classified pixels, so that the gradient signal is dominated by hard, misclassified examples. Per-class weights αc are set inversely proportional to class frequency but capped at an upper bound to prevent gradient explosion from extremely rare classes. The Dice loss component directly maximises the overlap between predicted and ground-truth masks on a per-class basis, which is particularly effective for small-object classes where pixel-level accuracy can be misleading. The total loss is a weighted sum: L_total = 0.7 × L_focal + 0.3 × L_dice, with the mixing coefficients chosen based on validation performance.

### Training Strategy

The training strategy evolved over 14 systematic experimental sessions. Early sessions (1–8) used aggressive class weights (up to 5000× for rare classes) and high learning rates (1e-3), which successfully forced the model to learn rare classes but caused catastrophic forgetting — previously well-performing classes (70–80% accuracy) collapsed to 15–40%. Middle sessions (9–11) explored oversampling and larger model variants, yielding temporary improvements that did not generalise.

The final strategy adopted a conservative fine-tuning protocol: moderate class weights (200–2500×) increased gradually across epochs, selective oversampling (8×) for the rarest classes only, and a very low learning rate (1e-5) for controlled parameter updates. Automated safety checks monitored per-class accuracy at every epoch and suspended training if any class dropped below 65%, preventing the runaway forgetting observed earlier. Training was capped at 40 epochs, with the best checkpoint selected based on validation loss.

### Prediction and Evaluation

At inference, the model produces a probability tensor P ∈ ℝ^(H × W × C) via softmax activation. The final segmentation mask is obtained by applying an argmax operation across the class dimension for each pixel: Ŷ(i,j) = argmax_c P(i,j,c). Model quality is assessed using per-class Intersection over Union (IoU), per-class pixel accuracy, and their macro-averages (mean IoU and mean accuracy), providing a comprehensive view of both dominant-class stability and rare-class learning progress.

## Implementation (ROS2 Setup)


The trained segmentation model was deployed within a ROS2 Humble Hawksbill pipeline to enable real-time scene understanding on the ROSWITHA platform. ROS2 was chosen over its predecessor for several reasons: its decentralised architecture eliminates the single-point-of-failure problem of the ROS1 master node, the Data Distribution Service (DDS) middleware provides built-in support for real-time operations with configurable Quality of Service (QoS) policies, and its native security features (encryption and authentication via SROS2) are essential for an assistive robot operating in human environments. The integration bridges the gap between offline model training and live robotic perception, addressing practical challenges around fault tolerance, resource management, and time synchronisation.
 
### System Architecture
 
The ROS2 implementation is structured into three modular nodes that communicate through a publish-subscribe model:
 
| Node                | Subscribes To       | Publishes To                  | Function                                                                 |
|---------------------|----------------------|-------------------------------|--------------------------------------------------------------------------|
| **Image Publisher**     | Camera hardware      | `/camera/rgb`                 | Captures RGB frames from the RealSense D435 camera or dataset and publishes them as `sensor_msgs/msg/Image` |
| **Inference Node**      | `/camera/rgb`        | `/segmentation/mask`          | Receives input images, preprocesses them (resize to 128×128, normalisation, RGB+XYZ channel augmentation), runs the trained `.keras` model in inference mode, and publishes per-pixel class predictions |
| **Visualisation Node**  | `/segmentation/mask` | `/segmentation/overlay`       | Converts predicted masks into colour-coded segmentation outputs and overlays them onto the original RGB image for human interpretation |
 
This decoupled design means each node can be developed, tested, and replaced independently. Multiple downstream consumers can subscribe to the same segmentation topic without modifying the inference node, and producers and consumers can operate at different rates.
 
### Communication and Synchronisation
 
**Quality of Service (QoS) Policies:** Camera streams use `SENSOR_DATA` QoS, which prioritises low latency over guaranteed delivery — under heavy load the system drops frames rather than building a backlog, keeping the perception clock current. Segmentation output uses `RELIABLE` QoS with a deeper queue, ensuring downstream consumers (e.g., navigation planners) receive every published mask even during brief network congestion.
 
**Message Synchronisation:** The RealSense D435 publishes RGB and depth streams at slightly different timestamps, introducing 10–30 ms timing variance. To handle this, a message filter with an approximate time synchroniser matches messages from both streams within a 50 ms tolerance window. The synchroniser maintains internal queues and pairs the closest-in-time RGB and depth frames, ensuring that the 6-channel input tensor fed to the model is temporally consistent.
 
### Data Flow Pipeline
 
The end-to-end execution within ROS2 follows this sequence:
 
```
Camera (D435) → RGB acquisition
             → Preprocessing (resize 128×128, normalise)
             → Channel augmentation (RGB + synthetic XYZ)
             → Model inference (forward pass through encoder-decoder)
             → Pixel-wise classification (argmax over 10 classes)
             → Post-processing and colour-coded visualisation
             → Published to ROS2 topics
```
 
The model processes each frame in approximately 65 ms on GPU (NVIDIA RTX 3090, ~15 Hz) and 180 ms on CPU (Intel i7-9700K, ~5.5 Hz), meeting the 10–15 Hz perception rate required for ROSWITHA's navigation and manipulation tasks.



```
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

source /opt/ros/humble/setup.bash

python3 -m venv ros2_env
source ros2_env/bin/activate

pip install tensorflow opencv-python numpy pillow labelme

cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python segmentation_node

cd ~/ros2_ws
colcon build
source install/setup.bash

ros2 run segmentation_node inference
```

---

## Results

| Class | IoU | Pixel Accuracy |
|------|------|----------------|
| Background | 0.839 | 0.9697 |
| Human | 0.814 | 0.6053 |
| Table | 0.824 | 0.5360 |
| Chair | 0.507 | 0.6862 |
| Robot | 0.729 | 0.3682 |
| Backpack | 0.343 | 0.2134 |
| Free Space | 0.883 | 0.5471 |
| Laptop | 0.172 | 0.0931 |
| Bottle | 0.638 | 0.3544 |
| Microwave | 0.001 | 0.0911 |
| **Mean IoU** | **0.575** | **0.4464** |


## Final comparisons

| Metric            | 7-Class Model | 10-Class Model |
|------------------|--------------|----------------|
| Global Accuracy  | 0.9584       | /              |
| Mean IoU         | 0.7270       | 0.575         |
| Mean Accuracy    | /            | 0.4464         |
| Number of Classes| 7            | 10             |

---

##  Limitations

- **Severe class imbalance** — Rare classes (backpack at 0.0035%) lack sufficient pixel-level training signal.
- **Catastrophic forgetting** — Aggressive balancing improved rare classes but degraded dominant ones by 15–40 percentage points.
- **Small object detection** — Laptops, bottles, and backpacks occupy few pixels and are often occluded.
- **Domain shift** — Mixing ROSWITHA and COCO datasets introduced inconsistent distributions.
- **ROS2 latency** — Real-time synchronisation between RGB and depth streams adds 10–30 ms jitter.

---

##  Future Work

The current mean IoU of 0.575 leaves significant room for improvement. The following directions, ordered by expected impact, target the three root causes: insufficient data, architectural limitations, and training inefficiency.
 
### 1. Dataset-Level Improvements (Highest Priority)
 
The single most impactful change would be expanding the training set for underrepresented classes. With backpack occupying only ~450,000 pixels (equivalent to roughly 27 full 128×128 images), no algorithmic technique can compensate for this data scarcity. Concrete steps include:
 
- **Targeted data collection** — Capture 2,000–5,000 additional annotated frames per rare class directly from ROSWITHA's environment, ensuring variety in object pose, lighting, and occlusion. This alone could push rare-class IoU from <0.2 to 0.4+ based on scaling-law studies in segmentation literature.
- **Synthetic data augmentation** — Use tools such as NVIDIA Omniverse or BlenderProc to generate photorealistic indoor scenes with controlled object placement. Synthetic-to-real transfer, combined with domain randomisation, has been shown to bridge the gap when real data is scarce.
- **Copy-paste augmentation** — Crop annotated object instances and paste them onto diverse background scenes during training. This is computationally cheap and directly increases effective pixel counts for rare classes without requiring new annotation.
- **Class-aware sampling** — Replace uniform epoch sampling with a strategy that samples each mini-batch to guarantee minimum representation of every class, ensuring the model sees rare objects in every training step.
 
### 2. Model-Level Improvements
 
- **Switch to a transformer-based backbone** — Architectures like SegFormer or Mask2Former capture long-range dependencies through self-attention, which helps detect small, scattered objects that convolution-only models miss. SegFormer in particular is lightweight enough for real-time inference.
- **Multi-scale feature fusion** — Integrate a Feature Pyramid Network (FPN) or HRNet-style parallel multi-resolution branches so that fine spatial details (critical for small objects) are preserved alongside high-level semantic features.
- **Dedicated small-object head** — Add a separate decoding branch at higher resolution (256×256 or 512×512) specifically for low-pixel classes, allowing the model to allocate capacity where it is most needed without compromising global segmentation.
- **Knowledge distillation** — Train a large teacher model offline and distil its predictions into the deployed lightweight model, retaining accuracy while meeting real-time constraints.
 
### 3. Training-Level Improvements
 
- **Curriculum learning** — Start training on well-represented classes, then progressively introduce rare classes. This reduces catastrophic forgetting by building a stable feature backbone before adapting to harder examples.
- **Elastic Weight Consolidation (EWC)** — Penalise changes to parameters that are important for previously learned classes, directly addressing the forgetting observed across our 14 training sessions.
- **Advanced loss functions** — Replace or augment focal loss with Lovász-Softmax loss, which directly optimises the IoU metric and has shown strong results under class imbalance.
- **Test-time augmentation (TTA)** — Apply multiple augmentations at inference and average predictions, boosting accuracy by 2–5% with no retraining cost.
 
### Estimated Impact
 
With the above improvements applied together — particularly a 10× increase in rare-class training data combined with a transformer backbone and curriculum learning — a realistic target is mIoU of 0.55–0.65, which would bring the 10-class model much closer to the 7-class baseline (0.727) while retaining the broader class coverage needed for real-world assistive robotics.



---

##  Conclusion

This work presented a complete end-to-end semantic segmentation framework for the assistive robot ROSWITHA, covering the full pipeline from raw annotated data to real-time ROS2 deployment. By extending perception from 7 to 10 classes — adding laptops, backpacks, and microwaves — the project broadened the robot's scene understanding for complex indoor environments. The FRA-UAS-SS architecture, combining an FC-HarDNet backbone with MMAF-Net adaptive fusion and SE attention blocks, demonstrated strong segmentation on dominant classes. 

The combined focal and Dice loss, paired with a conservative fine-tuning strategy developed over 14 systematic training sessions, achieved the best possible balance between rare-class learning and dominant-class stability. Successful deployment within ROS2 Humble at 15 Hz validated the practical feasibility of deep learning-based robotic perception. However, the mIoU drop from 0.7270 (7-class) to 0.575 (10-class) confirms that for classes below 0.01% of training pixels, algorithmic techniques alone cannot compensate for data scarcity. Future work should prioritise targeted data collection, transformer-based architectures, and continual learning techniques to bridge this gap
