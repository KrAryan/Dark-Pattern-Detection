from ultralytics import YOLO
import os
model = YOLO("yolov8n.yaml")
model.train(
    data="Data/image/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="dark_pattern_detector",
    pretrained=False  # ✅ Training from scratch
)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/dark_pattern_detector.pt")
print("✅ Training complete. Model saved to models/dark_pattern_detector.pt")
