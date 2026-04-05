# 💸 Multi-Currency Banknote Recognition AI (YOLOv8 & ResNet50)

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-green.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

A complete, high-performance **Deep Learning Pipeline** for real-time banknote detection and denomination classification. This project integrates cutting-edge Vision models to accurately recognize multi-national currency (Vietnam Dong, Indian Rupee, and Thai Baht) even under heavy occlusion, varying angles, or complex backgrounds.

---

## 🌟 How It Works (The 2-Tier Architecture)

To maximize accuracy and bypass dataset inconsistencies (e.g., bounding box variations between holistic note borders vs. specific numerical features), this project leverages a dual-network approach:
1. **Tier 1 - Object Detection (YOLOv8):** Scans the entire frame, locating any visible banknote and cropping its precise Coordinates (Bounding Box).
2. **Tier 2 - Image Classification (ResNet50):** Processes the isolated crops from Tier 1 through a robust Transfer Learning classifier to determine the exact currency denomination and country of origin.

---

## 🚀 Features
- **Automated Data Processing Pipeline:** Python scripts to cleanly parse disjoint datasets, normalize formats into YOLO-friendly metrics, and automate ImageFolder structuring.
- **Auto-Balanced Splitting:** Dynamically splits classes (Train:Val:Test - 80:10:10).
- **Automated Reporting:** Generates loss curves, confusion matrices, and detailed JSON metrics dynamically upon ResNet completion.
- **Real-Time Webcam GUI (`app.py`):** A sleek, fully interactive Streamlit application to invoke Real-Time OpenCV streaming seamlessly with the compiled `.pt` and `.pth` weight files.

---

## 📂 Repository Structure

```text
├── app.py                     # Giao diện Web Streamlit Nhận diện Tiền Realtime
├── inference.py               # Script Test nghiệm thu tĩnh trên từng hình ảnh
├── prepare_yolo_data.py       # Script dọn dẹp & hợp nhất Dataset cho YOLO
├── prepare_resnet_data.py     # Script tự động Crop nhãn để đóng khuôn Dataset cho ResNet
├── train_yolo.py              # Script huấn luyện YOLOv8 kèm Early Stopping
├── train_resnet.py            # Script huấn luyện ResNet50 kèm Sinh Biểu Đồ
├── .gitignore               
└── README.md                  # Hướng dẫn chi tiết
```

---

## ⚙️ Installation & Usage

### 1. Prerequisites
Ensure you have `Python 3.10` environments configured.
```bash
# Clone the repository


# Install core dependencies
pip install torch torchvision ultralytics opencv-python Pillow scikit-learn matplotlib seaborn streamlit
```

### 2. Dataset Preparation
Place your raw Datasets into the root directory (e.g., `Dataset_VND`, `dataset_inr`, `dataset_thb`).
Execute the Data Engineering scripts to unify classes and map subsets properly.
```bash
python prepare_yolo_data.py
python prepare_resnet_data.py
```
*Note: This generates two cleaned directories (`yolo_merged_dataset/` and `ResNet_Data/`), isolating them perfectly for our two respective neural networks.*

### 3. Training the Models
**Phase 1: Train Object Detector**
```bash
python train_yolo.py
```

**Phase 2: Train Denomination Classifier**
```bash
python train_resnet.py
```
*(Performance metrics, F1-Scores, and plots will automatically be written to disk.)*

---

## 🔍 Command-Line Inference

To test the trained models on a single image without launching the Web UI, you can use the command-line inference script:
```bash
python inference.py <path_to_your_image.jpg>
```
The script will perform the full 3-stage pipeline (YOLOv8 Localization -> Extract ROI -> ResNet50 Classification) and output an annotated image named `inference_result.jpg` in your current directory.

---

## 📸 Real-Time Application 

Once you compile the weights (`best.pt`, `best_resnet.pth`, and `resnet_classes.txt`), bring them to the root repository folder.

Spin up the elegant UI instantly via Streamlit to demonstrate actual capability via Laptop Camera:
```bash
streamlit run app.py
```
Toggle the "START AI ANALYSIS" checkbox on the left sidebar, showcase a banknote, and watch the system instantly overlay the denomination in bright green bounding boxes.

---

## 🤝 Contribution
Contributions, bug reports, and optimizations are heavily welcomed! Feel free to raise an Issue or submit a Pull Request.

**Author:** [NDHUY]  
**License:** MIT License  
*(This project was formulated and optimized as a deep dive into Computer Vision Pipelines!)*
