import os
import cv2
import pytesseract
import joblib
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from ultralytics import YOLO

# === Setup ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "Data")
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# === Load YOLOv8 Model ===
yolo_path = os.path.join(MODEL_DIR, "yolov8_darkpattern.pt")
if not os.path.isfile(yolo_path):
    raise FileNotFoundError("YOLOv8 model not found at: models/yolov8_darkpattern.pt")
yolo_model = YOLO(yolo_path)

# === Read URLs ===
url_file = os.path.join(DATA_DIR, "cleaned_landing_pages.txt")
with open(url_file, "r") as f:
    urls = [line.strip() for line in f if line.strip()]

# === Setup Selenium Headless Chrome ===
options = Options()
options.add_argument("--headless")
options.add_argument("--window-size=1920,1080")
driver = webdriver.Chrome(options=options)

# === Screenshot + Prediction ===
results = []

for url in urls:
    try:
        driver.get(url)
        name = url.split("//")[1].replace(".", "_").replace("/", "_")
        shot_path = os.path.join(SCREENSHOT_DIR, f"{name}.png")
        driver.save_screenshot(shot_path)

        # YOLO predict
        preds = yolo_model.predict(shot_path, conf=0.25)[0]
        labels = []
        if preds.boxes:
            labels = [preds.names[int(i)] for i in preds.boxes.cls.cpu().numpy()]

        # OCR (Optional)
        img = cv2.imread(shot_path)
        text = pytesseract.image_to_string(img)

        results.append({
            "url": url,
            "screenshot": shot_path,
            "dark_patterns_detected": labels,
            "ocr_text": text[:300].replace("\n", " ")  # shorten
        })

        print(f"[✓] {url} done")

    except Exception as e:
        print(f"[!] Error processing {url}: {e}")

driver.quit()

# === Save CSV ===
df = pd.DataFrame(results)
df.to_csv(os.path.join(RESULT_DIR, "darkpattern_yolo_results.csv"), index=False)
print("[✓] All done. Results saved.")
