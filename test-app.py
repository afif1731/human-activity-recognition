import time
import cv2
import torch
import joblib
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from torchvision import models, transforms
from torch import nn
import torch.nn.functional as F
from PIL import Image
import os
import psutil

# -------- CONFIG --------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
YOLO_WEIGHTS = "./model/yolo12n.pt"
CLASSIFIER_WEIGHTS = "./model/best_densenet_yolo.pth"
LABEL_ENCODER_PATH = "./model/label_encoder.pkl"
CONF_THRESH = 0.5
LABEL_CONF_THRESH = 0.5
IMG_SIZE = 224
YOLO_IMGZ = 320
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS = 2
CPU_CORES = os.cpu_count() or 1

current_mode = "Working" # 'Working' or 'Break'
working_labels = {"sitting", "using_laptop", "drinking", "listening_to_music"}
# ------------------------

yolo = YOLO(YOLO_WEIGHTS)

le_path = Path(LABEL_ENCODER_PATH)
if le_path.exists():
    le = joblib.load(str(le_path))
    class_names = list(le.classes_)
    print(f"Loaded label encoder with {len(class_names)} classes.")
else:
    class_names = [f"class_{i}" for i in range(2)]  # fallback
    print("⚠️ Label encoder not found — using fallback numeric labels.")
num_classes = len(class_names)

densenet_best = models.densenet121(weights=None)
num_features = densenet_best.classifier.in_features
densenet_best.classifier = nn.Linear(num_features, num_classes)
state_dict = torch.load(CLASSIFIER_WEIGHTS, map_location=DEVICE)
densenet_best.load_state_dict(state_dict)
densenet_best.to(DEVICE)
densenet_best.eval()
print("Loaded DenseNet classifier weights.")

to_tensor = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

def draw_label(img, text, x, y, color=(0,255,0)):
    if not text:
        return
    (w, h), _ = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
    cv2.rectangle(img, (x, y - h - 6), (x + w + 6, y + 6), (0,0,0), -1)
    cv2.putText(img, text, (x + 3, y), FONT, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam (index 0).")

pid = os.getpid()
process = psutil.Process(pid)
process.cpu_percent(interval=None)

print("Press 'm' to switch mode. Press 'q' to quit.")
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = yolo(rgb_frame, imgsz=YOLO_IMGZ, conf=CONF_THRESH, verbose=False)
    if results and len(results) > 0:
        res = results[0]
        if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)

            valid_indices = [i for i, c in enumerate(classes) if c == 0]
            if not valid_indices:
                pass 
            else:
                best_idx = max(valid_indices, key=lambda i: confs[i])
                x1, y1, x2, y2 = map(int, xyxy[best_idx])
                det_conf = float(confs[best_idx])

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_w, x2), min(frame_h, y2)

                if current_mode == "Working":
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
                        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        inp = to_tensor(pil_crop).unsqueeze(0).to(DEVICE)

                        with torch.no_grad():
                            logits = densenet_best(inp)
                            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                            top_idx = int(np.argmax(probs))
                            top_conf = float(probs[top_idx])

                        label = class_names[top_idx] if top_idx < len(class_names) else str(top_idx)
                        
                        display_text = ""
                        color = (0, 0, 255) # Red

                        if top_conf < LABEL_CONF_THRESH:
                            display_text = f"Working (Low Conf: {top_conf:.2f})"
                            color = (0, 255, 0) # Green
                        elif label in working_labels:
                            display_text = f"Working: {label} ({top_conf:.2f})"
                            color = (0, 255, 0) # Green
                        else:
                            display_text = f"Distracted: {label} ({top_conf:.2f})"
                            color = (0, 0, 255) # Red
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        draw_label(frame, display_text, x1, y1 - 4, color=color)
                
                elif current_mode == "Break":
                    color = (0, 0, 255) # Red
                    display_text = "Working"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    draw_label(frame, display_text, x1, y1 - 4, color=color)


    cur_time = time.time()
    fps = 1.0 / (cur_time - prev_time) if (cur_time - prev_time) > 0 else 0.0
    prev_time = cur_time

    cpu_usage = process.cpu_percent(interval=None)
    ram_usage_mb = process.memory_info().rss / (1024 * 1024)

    draw_label(frame, f"Mode: {current_mode}", 10, frame_h - 100, color=(255,255,0))
    draw_label(frame, f"CPU: {cpu_usage:.1f} %. AVG: {(cpu_usage / CPU_CORES):.1f} %", 10, frame_h - 70, color=(255,255,0))
    draw_label(frame, f"RAM: {ram_usage_mb:.1f} MB", 10, frame_h - 40, color=(255,255,0))
    draw_label(frame, f"FPS: {fps:.1f}", 10, frame_h - 10, color=(255,255,0))

    cv2.imshow("Realtime Recognition (Single Object)", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        current_mode = "Break" if current_mode == "Working" else "Working"
        print(f"Mode switched to: {current_mode}")

cap.release()
cv2.destroyAllWindows()