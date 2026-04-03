from ultralytics import YOLO

def main():
    print("Khởi tạo mô hình YOLOv8n (Nano) làm nền tảng. Bạn có thể đổi sang yolov8s.pt nếu Server mạnh...")
    model = YOLO("yolov8n.pt")  

    print("Bắt đầu huấn luyện YOLOv8... Quá trình này sẽ sử dụng GPU nếu có sẵn.")
    results = model.train(
        data="yolo_merged_dataset/dataset.yaml",
        epochs=100,           # Chạy tối đa 100 epochs để tối ưu
        patience=20,          # Cơ chế Early Stopping nếu mô hình không cải thiện sau 20 epochs
        imgsz=640,            # Resize ảnh về 640x640 (chuẩn của Yolo)
        batch=16,             # Batch size
        project="runs/detect",
        name="train_yolo_model",
        plots=True            # Bật cờ này để Yolo tự sinh đồ thị PR Curve, F1, Loss Curve
    )

    print("Hoàn tất huấn luyện YOLO! Trọng số tốt nhất được lưu tại: runs/detect/train_yolo_model/weights/best.pt")

if __name__ == '__main__':
    main()
