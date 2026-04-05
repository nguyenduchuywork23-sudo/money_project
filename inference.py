import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO
import sys

def predict_money(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting inference on device: {device}...")

    # Stage 1: Object Detection (YOLOv8)
    try:
        yolo_model = YOLO(r"C:\Users\Nguye\OneDrive\Desktop\money_project\money_project\runs\detect\runs\detect\train_yolo_model3\weights\best.pt")
    except Exception as e:
        print(f"\n[ERROR] YOLO model not found: {e}")
        return

    # Stage 3: Image Classification (ResNet50)
    try:
        with open("resnet_classes.txt", "r") as f:
            class_names = f.read().split(",")
        
        num_classes = len(class_names)
        resnet_model = models.resnet50(weights=None)
        num_ftrs = resnet_model.fc.in_features
        resnet_model.fc = nn.Linear(num_ftrs, num_classes)
        resnet_model.load_state_dict(torch.load("best_resnet.pth", map_location=device, weights_only=True))
        resnet_model = resnet_model.to(device)
        resnet_model.eval()
    except Exception as e:
        print(f"\n[ERROR] ResNet model or class names not found: {e}")
        return
        
    resnet_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Read input image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[ERROR] Cannot read image from path: {image_path}")
        return
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # ---------------------------
    # STAGE 1: LOCALIZATION
    # ---------------------------
    print("Running bounding box extraction...")
    results = yolo_model(img_rgb, verbose=False)
    boxes = results[0].boxes
    
    if len(boxes) == 0:
        print("-> No target objects detected.")
    
    # ---------------------------
    # STAGE 2 & 3: ROI CROPPING & CLASSIFICATION
    # ---------------------------
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        
        # Stage 2: ROI Extraction
        crop_pil = pil_img.crop((x1, y1, x2, y2))
        
        # Stage 3: Feature Classification
        input_tensor = resnet_transforms(crop_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = resnet_model(input_tensor)
            _, preds = torch.max(outputs, 1)
            predicted_class = class_names[preds[0].item()]
            
        # Annotate origin image
        label = f"{predicted_class} ({conf:.2f})"
        print(f"-> Detected object: {label}")
        
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        (c_w, c_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Prevent labels from rendering out-of-bounds
        if y1 - c_h - 10 < 0:
            bg_y1 = y1
            bg_y2 = y1 + c_h + 10
            text_y = y1 + c_h + 5
        else:
            bg_y1 = y1 - c_h - 10
            bg_y2 = y1
            text_y = y1 - 5
            
        cv2.rectangle(img_bgr, (x1, bg_y1), (x1 + c_w, bg_y2), (0, 0, 0), -1) 
        cv2.putText(img_bgr, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out_path = "inference_result.jpg"
    cv2.imwrite(out_path, img_bgr)
    print(f"Results successfully saved to: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <dataset_image.jpg>")
    else:
        predict_money(sys.argv[1])
