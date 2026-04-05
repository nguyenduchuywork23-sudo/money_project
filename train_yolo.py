from ultralytics import YOLO

def main():
    print("Initializing YOLOv8n (Nano) backbone model...")
    model = YOLO("yolov8n.pt")  

    print("Starting YOLOv8 training. GPU acceleration will be used if available.")
    results = model.train(
        data="yolo_merged_dataset/dataset.yaml",
        epochs=100,           # Maximum number of epochs for optimization
        patience=20,          # Early stopping patience parameter
        imgsz=640,            # Standard input image size for YOLO
        batch=16,             # Mini-batch size
        project="runs/detect",
        name="train_yolo_model",
        plots=True            # Generate evaluation metrics plots (PR Curve, F1, Loss)
    )

    print("Training complete. Best weights saved to: runs/detect/train_yolo_model/weights/best.pt")

if __name__ == '__main__':
    main()
