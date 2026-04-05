import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO
import numpy as np
import os

# Base page configuration
st.set_page_config(page_title="Banknote Recognition System", layout="centered")

# Custom CSS for modern UI design (Glassmorphism & Gradient)
st.markdown("""
<style>
    /* Global dark background */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Header styling */
    h1 {
        color: #38bdf8;
        font-family: 'Inter', sans-serif;
        text-align: center;
        font-weight: 800;
        text-shadow: 0px 4px 10px rgba(56, 189, 248, 0.3);
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-family: 'Inter', sans-serif;
        margin-bottom: 2rem;
    }
    
    /* Gradient Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #0ea5e9, #2563eb);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 700;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.4);
        transform: translateY(-2px);
    }
    
    /* Result card (Glassmorphism) */
    .result-card {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 25px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        margin-top: 30px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("Multi-Currency Recognition")
st.markdown("<p class='subtitle'>Multi-Currency Banknote Recognition System (VND, INR, THB)<br>AI Object Detection & Classification</p>", unsafe_allow_html=True)

# -------------------------------
# MODEL LOADING (CACHE)
# -------------------------------
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load YOLOv8 Model
    yolo_path = os.path.join("runs", "detect", "runs", "detect", "train_yolo_model3", "weights", "best.pt")
    yolo_model = None
    if os.path.exists(yolo_path):
        yolo_model = YOLO(yolo_path)
    else:
        st.error(f"YOLO model not found at: {yolo_path}")

    # 2. Load ResNet50 Model
    resnet_model = None
    class_names = []
    try:
        with open("resnet_classes.txt", "r") as f:
            class_names = f.read().split(",")
        num_classes = len(class_names)
        
        resnet_model = models.resnet50(weights=None)
        num_ftrs = resnet_model.fc.in_features
        resnet_model.fc = nn.Linear(num_ftrs, num_classes)
        resnet_model.load_state_dict(torch.load("best_resnet.pth", map_location=device))
        resnet_model = resnet_model.to(device)
        resnet_model.eval()
    except Exception as e:
        st.error(f"Error loading ResNet model or classes: {e}")
        
    return yolo_model, resnet_model, class_names, device

yolo, resnet, classes, dev = load_models()

resnet_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -------------------------------
# FRONTEND INTERFACE
# -------------------------------
uploaded_file = st.file_uploader("Upload banknote image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    cv_img = np.array(image)
    cv_img = cv_img[:, :, ::-1].copy() # Convert RGB -> BGR for OpenCV processing

    # Display original image
    st.image(image, caption="Original Image", use_container_width=True)

    if st.button("START AI ANALYSIS"):
        if yolo is None or resnet is None:
            st.error("System is not ready due to missing model weights. Please check the terminal.")
        else:
            with st.spinner("Running YOLO and ResNet parallel inference..."):
                # --- STAGE 1: LOCALIZATION ---
                results = yolo(image, verbose=False)
                boxes = results[0].boxes
                
                if len(boxes) == 0:
                    st.warning("YOLO Detection: No banknote features detected in the image.")
                else:
                    # --- STAGE 2 & 3: ROI CROPPING & CLASSIFICATION ---
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        
                        # Extract Region of Interest (ROI)
                        crop_pil = image.crop((x1, y1, x2, y2))
                        input_tensor = resnet_transforms(crop_pil).unsqueeze(0).to(dev)
                        
                        # ResNet50 Classification
                        with torch.no_grad():
                            outputs = resnet(input_tensor)
                            _, preds = torch.max(outputs, 1)
                            predicted_class = classes[preds[0].item()]
                            
                        # Draw Red Bounding Box
                        label = f"{predicted_class} ({conf:.2f})"
                        cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 0, 255), 3) 
                        
                        # Calculate text offset
                        (c_w, c_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        
                        if y1 - c_h - 10 < 0:
                            bg_y1 = y1
                            bg_y2 = y1 + c_h + 10
                            text_y = y1 + c_h + 5
                        else:
                            bg_y1 = y1 - c_h - 10
                            bg_y2 = y1
                            text_y = y1 - 5
                            
                        # Draw black background and green text
                        cv2.rectangle(cv_img, (x1, bg_y1), (x1 + c_w, bg_y2), (0, 0, 0), -1) 
                        cv2.putText(cv_img, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    st.balloons()
                    
                    # Convert BGR (OpenCV) back to RGB (Streamlit/Web)
                    final_img = cv_img[:, :, ::-1]
                    
                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    st.success(f"**ANALYSIS COMPLETE!** Detected **{len(boxes)}** object(s).")
                    st.image(final_img, caption="Localization & Classification Results", use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
