# 🧠 Semantic Segmentation using RGB-D with ROS2 Integration

## 📌 Introduction
This project presents an end-to-end semantic segmentation pipeline integrated with ROS2 for real-time robotic perception.

## ⚙️ Technologies Used
- Python, TensorFlow, OpenCV, NumPy
- ROS2 (Humble), Colcon
- LabelMe, TFRecords

## 🏗️ Architecture
Dataset → JSON → Masks → TFRecords → Training → Model → ROS2 → Inference

## 🧪 Methodology
- JSON to mask conversion
- RGB + XYZ input (6-channel)
- TFRecord pipeline
- UNet-based training
- Argmax prediction

## 🤖 Implementation (ROS2 Setup)

### Workspace
mkdir -p ~/ros2_ws/src  
cd ~/ros2_ws  

### Source ROS2
source /opt/ros/humble/setup.bash  

### Virtual Environment
python3 -m venv ros2_env  
source ros2_env/bin/activate  

### Install Dependencies
pip install tensorflow opencv-python numpy pillow labelme  

### Create Package
ros2 pkg create --build-type ament_python segmentation_node  

### Build
colcon build  
source install/setup.bash  

### Run
ros2 run segmentation_node inference  

## 📊 Results
- Good performance on major classes
- Real-time ROS2 inference
- Stable pipeline

## ⚠️ Limitations
- Weak small object detection
- Class imbalance issues
- Latency in ROS2

## 🚀 Future Work
- Improve dataset
- Optimize model
- Enhance real-time performance

## 📁 Structure
project/
├── data/
├── tfrecords/
├── models/
├── ros2_ws/
└── README.md

## ✅ Conclusion
Complete pipeline from training to ROS2 deployment achieved successfully.
