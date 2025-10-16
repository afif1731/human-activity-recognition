# Real-time recognition (YOLOv12 + ResNet18)
# Only detect and classify ONE object (highest confidence) at a time.

import time
import cv2
import torch
import joblib
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from torchvision import models, transforms
import torch.nn.functional as F

# -------- CONFIG --------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
YOLO_WEIGHTS = "./model/yolo12n.pt"
CLASSIFIER_WEIGHTS = "./model/best_resnet_yolo.pth"
LABEL_ENCODER_PATH = "./model/label_encoder.pkl"
CONF_THRESH = 0.6
LABEL_CONF_THRESH = 0.85
IMG_SIZE = 224
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS = 2
# ------------------------

# Load YOLO model (for detection)
yolo = YOLO(YOLO_WEIGHTS)

# Load label encoder
class_names = None
le_path = Path(LABEL_ENCODER_PATH)
if le_path.exists():
    le = joblib.load(str(le_path))
    class_names = list(le.classes_)
    num_classes = len(class_names)
    print(f"Loaded label encoder with {num_classes} classes.")
else:
    class_names = [str(i) for i in range(2)]
    num_classes = len(class_names)
    print("Label encoder not found — using fallback numeric labels.")

# Load classifier (ResNet101)
resnet = models.resnet101(pretrained=False)
resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)
resnet.load_state_dict(torch.load(CLASSIFIER_WEIGHTS, map_location=DEVICE))
resnet.to(DEVICE)
resnet.eval()
print("Loaded classifier weights.")

# Transform for classifier input
to_tensor = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Utility to draw text with background
def draw_label(img, text, x, y, color=(0,255,0)):
    (w,h), _ = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
    cv2.rectangle(img, (x, y - h - 6), (x + w + 6, y + 6), (0,0,0), -1)
    cv2.putText(img, text, (x + 3, y), FONT, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam (index 0).")

print("Press 'q' to quit.")

prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo(frame, imgsz=640, conf=CONF_THRESH, verbose=False)
    if results and len(results) > 0:
        res = results[0]
        if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
            # Extract all detections
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)

            # Filter: optional — only keep "person" (COCO class 0)
            valid_indices = [i for i, c in enumerate(classes) if c == 0]  # only person
            if not valid_indices:
                valid_indices = list(range(len(confs)))  # fallback: all classes

            # Pick highest-confidence detection only
            best_idx = max(valid_indices, key=lambda i: confs[i])
            x1, y1, x2, y2 = map(int, xyxy[best_idx])
            det_conf = float(confs[best_idx])

            # Clamp box
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                inp = to_tensor(crop).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits = resnet(inp)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                    top_idx = int(np.argmax(probs))
                    top_conf = float(probs[top_idx])

                label = class_names[top_idx] if top_idx < len(class_names) else str(top_idx)
                if top_conf < LABEL_CONF_THRESH:
                    label_text = f""
                else:
                    label_text = f"{label}: {top_conf:.2f}"

                # Draw box and label
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                draw_label(frame, label_text, x1, y1 - 4, color=color)

    # FPS display
    cur_time = time.time()
    fps = 1.0 / (cur_time - prev_time) if (cur_time - prev_time) > 0 else 0.0
    prev_time = cur_time
    draw_label(frame, f"FPS: {fps:.1f}", 10, 30, color=(255,255,0))

    cv2.imshow("Realtime Recognition (Single Object)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
