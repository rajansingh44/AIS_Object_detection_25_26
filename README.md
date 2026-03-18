# Semantic Segmentation for ROSWITHA using RGB-D and ROS2

---

##  Introduction

This project focuses on enhancing the visual perception capabilities of the assistive robot **ROSWITHA (RObot System WITH Autonomy)** by implementing a deep learning-based **10-class semantic segmentation model**.

The system enables real-time scene understanding in complex indoor environments by combining **RGB and depth (XYZ) data**. The model extends a previous 7-class segmentation framework by introducing three new object classes:

- Laptop  
- Backpack  
- Microwave  

However, the extension introduces **severe class imbalance challenges**, where rare classes (e.g., backpack: 0.0035%) are significantly underrepresented, leading to training instability and **catastrophic forgetting**.

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

```
RGB + JSON → Mask Generation → TFRecords
          → Training → Model (.keras)
          → ROS2 Node → Inference → Output
```

---

## 🧪 Methodology

### Data Preparation
- JSON annotations converted to pixel masks  
- Label normalization and filtering  

### Multi-Modal Input
```
Input: H × W × 6 (RGB + XYZ)
```

### Model Output
```
P ∈ R^(H × W × C)
Ŷ(i,j) = argmax(P(i,j,c))
```

---

## Implementation (ROS2 Setup)

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

| Class | IoU |
|------|------|
| Background | 0.9245 |
| Human | 0.5200 |
| Table | 0.4140 |
| Chair | 0.4519 |
| Robot | 0.3048 |
| Backpack | 0.1609 |
| Free Space | 0.5050 |
| Laptop | 0.0663 |
| Bottle | 0.2268 |
| Microwave | 0.0673 |
| **Mean IoU** | **0.3642** |

---

##  Limitations

- Poor detection of small objects  
- Severe class imbalance  
- Catastrophic forgetting  
- ROS2 latency and synchronization issues  

---

##  Future Work

- Improve dataset balance  
- Optimize model  
- Enhance real-time performance  

---

##  Project Structure

```
project/
├── data/
├── tfrecords/
├── models/
├── ros2_ws/
└── README.md
```

---

## ✅ Conclusion

This project demonstrates a complete pipeline from training to ROS2 deployment, highlighting both strengths and real-world challenges in semantic segmentation for robotics.
