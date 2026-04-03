import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO
import sys

def predict_money(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Bắt đầu Inference trên {device}...")

    # Load YOLO Model (Tầng 1)
    try:
        yolo_model = YOLO("/srv/nns/users/nmhung/vehicle-detection/cash_project/money_project/runs/detect/runs/detect/train_yolo_model3/weights/best.pt")
    except Exception as e:
        print("\n[LỖI] Không tìm thấy model YOLO. Có thể YOLO chưa được Train xong.")
        print("Hãy chạy 'python train_yolo.py' trước!")
        return

    # Load ResNet Model (Tầng 3)
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
        print("\n[LỖI] Không tìm thấy 'best_resnet.pth' hoặc file 'resnet_classes.txt'.")
        print("Hãy đảm bảo bạn đã chạy 'python train_resnet.py' thành công!")
        return
        
    resnet_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Đọc ảnh đầu vào
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[LỖI] Không thể đọc ảnh: {image_path}")
        return
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # ---------------------------
    # TẦNG 1: DÙNG YOLO TÌM VỊ TRÍ
    # ---------------------------
    print("Mô hình YOLO đang chạy để trích xuất vật mẫu...")
    results = yolo_model(img_rgb, verbose=False)
    boxes = results[0].boxes
    
    if len(boxes) == 0:
        print("--> YOLO không tìm thấy đồng tiền hoặc đặc điểm nhận dạng nào trên ảnh này.")
    
    # ---------------------------
    # TẦNG 2 & 3: CẮT ẢNH & ĐOÁN MỆNH GIÁ
    # ---------------------------
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        
        # Tầng 2: Cắt ảnh (Crop theo ROI)
        crop_pil = pil_img.crop((x1, y1, x2, y2))
        
        # Tầng 3: ResNet50
        input_tensor = resnet_transforms(crop_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = resnet_model(input_tensor)
            _, preds = torch.max(outputs, 1)
            predicted_class = class_names[preds[0].item()]
            
        # Vẽ Khung vào Hình gốc
        label = f"{predicted_class} ({conf:.2f})"
        print(f"-> Đã quét trúng: {label}")
        
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3) # Khung viền Đỏ
        
        # Đo kích thước chữ
        (c_w, c_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Chống lỗi văng chữ ra ngoài màn hình nếu Box nằm quá sát lề trên
        if y1 - c_h - 10 < 0:
            bg_y1 = y1
            bg_y2 = y1 + c_h + 10
            text_y = y1 + c_h + 5
        else:
            bg_y1 = y1 - c_h - 10
            bg_y2 = y1
            text_y = y1 - 5
            
        # Nền đen cho chữ
        cv2.rectangle(img_bgr, (x1, bg_y1), (x1 + c_w, bg_y2), (0, 0, 0), -1) 
        # Cập nhật Text xanh lá cuốn hút
        cv2.putText(img_bgr, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out_path = "inference_result.jpg"
    cv2.imwrite(out_path, img_bgr)
    print(f"\n[THÀNH CÔNG] Đã lưu kết quả tại: {out_path}")
    print("Bạn có thể copy file này ra máy tính mở lên để chiêm ngưỡng.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách sử dụng (Gõ trên Terminal):")
        print("python inference.py <tên_file_ảnh.jpg>")
    else:
        predict_money(sys.argv[1])
