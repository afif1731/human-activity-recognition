import time
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from torchvision import models, transforms
from torch import nn
import torch.nn.functional as F
from PIL import Image
import os
import psutil
import threading
import queue
from collections import deque

# -------- CONFIG --------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

YOLO_WEIGHTS = "./model/yolo11n.pt"
MODEL_WEIGHTS = "./model/mobilenet_lstm_har.pth"
LABEL_PATH = "./model/class_names.pth"

CONF_THRESH = 0.25
LABEL_CONF_THRESH = 0.6
IMG_SIZE = 224
YOLO_IMGSZ = 320

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS = 2
CPU_CORES = os.cpu_count() or 1

DETECT_EVERY_N_FRAMES = 5 # Lakukan deteksi setiap N frame (untuk performa)
HISTORY_SIZE = 5          # Smoothing hasil prediksi

# Mode kerja
current_mode = "Working" 
# Sesuaikan himpunan ini dengan nama kelas yang ada di dataset Anda
working_labels = {"sitting", "using_laptop", "writing", "reading"} 

# Shared Variables untuk Threading
latest_result = {
    "box": None,
    "display_text": "",
    "color": (0, 0, 0),
    "found": False
}
result_lock = threading.Lock()
frame_queue = queue.Queue(maxsize=1)
running = True

detection_history = deque(maxlen=HISTORY_SIZE)

# ------------------------
# 1. DEFINISI ARSITEKTUR MODEL (Harus sama persis dengan Training)
# ------------------------
class MobileNetHAR(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetHAR, self).__init__()
        self.base_model = models.mobilenet_v3_large(weights=None)
        
        self.features = self.base_model.features
        self.avgpool = self.base_model.avgpool
        
        input_features = 960 
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_features, 1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

# ------------------------
# 2. INIT & LOAD MODELS
# ------------------------
print(f"Loading models on {DEVICE}...")

# A. Load YOLO
try:
    yolo = YOLO(YOLO_WEIGHTS)
except Exception as e:
    print(f"Error loading YOLO: {e}. Downloading default yolo11n.pt...")
    yolo = YOLO("yolo11n.pt")

# B. Load Label Encoder (List of Strings)
if os.path.exists(LABEL_PATH):
    class_names = torch.load(LABEL_PATH)
    print(f"Loaded {len(class_names)} classes: {class_names}")
else:
    print(f"⚠️ Label file not found at {LABEL_PATH}! Creating dummy classes.")
    class_names = ["Unknown"] # Fallback

num_classes = len(class_names)

# C. Load Classification Model
har_model = MobileNetHAR(num_classes=num_classes)

if os.path.exists(MODEL_WEIGHTS):
    state_dict = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
    har_model.load_state_dict(state_dict)
    print("HAR Model weights loaded successfully.")
else:
    print(f"⚠️ Model weights not found at {MODEL_WEIGHTS}!")

har_model.to(DEVICE)
har_model.eval()

# Transformasi (Sama seperti Validation/Test saat training)
to_tensor = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ------------------------
# 3. HELPER FUNCTIONS
# ------------------------
def draw_label(img, text, x, y, color=(0,255,0)):
    if not text: return
    (w, h), _ = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
    # Background hitam untuk teks agar terbaca
    cv2.rectangle(img, (x, y - h - 6), (x + w + 6, y + 6), (0,0,0), -1)
    cv2.putText(img, text, (x + 3, y), FONT, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)

def determine_final_status(history):
    if not history: return "Working", (0, 255, 0)
    
    # Simple majority voting atau thresholding
    distracted_count = sum(1 for status in history if status == "Distracted")
    threshold = len(history) // 2 + 1 

    if distracted_count >= threshold:
        return "Distracted", (0, 0, 255) # Red
    return "Working", (0, 255, 0) # Green

# ------------------------
# 4. WORKER THREAD (INFERENCE)
# ------------------------
def inference_worker():
    global latest_result, detection_history
    
    while running:
        try:
            # Ambil frame dari queue (blocking max 0.1s)
            frame_data = frame_queue.get(timeout=0.1) 

            frame_rgb = frame_data['img'] # Format PIL/RGB numpy
            frame_h, frame_w = frame_data['size']
            mode_at_capture = frame_data['mode']

            # --- A. DETEKSI OBJEK (YOLO) ---
            # Cari hanya class 0 (person)
            results = yolo(frame_rgb, imgsz=YOLO_IMGSZ, conf=CONF_THRESH, classes=[0], verbose=False)
            
            found_person = False
            best_box = None
            max_area = 0
            
            # --- LOGIKA BARU: CROP BERDASARKAN AREA TERBESAR ---
            if results:
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        area = (x2 - x1) * (y2 - y1)
                        
                        # Pilih bounding box dengan area terbesar
                        if area > max_area:
                            max_area = area
                            best_box = [int(x1), int(y1), int(x2), int(y2)]

            if best_box is not None:
                x1, y1, x2, y2 = best_box
                # Pastikan koordinat dalam range
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_w, x2), min(frame_h, y2)
                found_person = True

            # --- B. KLASIFIKASI AKTIVITAS ---
            if found_person and mode_at_capture == "Working":
                crop = frame_rgb[y1:y2, x1:x2]
                
                # Validasi ukuran crop
                if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
                    pil_crop = Image.fromarray(crop)
                    
                    # Preprocess & Inference
                    inp = to_tensor(pil_crop).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        output = har_model(inp)
                        probs = F.softmax(output, dim=1).cpu().numpy()[0]
                        top_idx = int(np.argmax(probs))
                        top_conf = float(probs[top_idx])

                    # Mapping ke nama kelas
                    raw_label = class_names[top_idx] if top_idx < len(class_names) else "Unknown"
                    display_label = raw_label

                    # Logika Status
                    status_result = "Working"
                    if top_conf < LABEL_CONF_THRESH:
                        display_label = f"{raw_label} (?)"
                        # Jika ragu, asumsikan status sebelumnya (opsional) atau tetap Working
                    elif raw_label in working_labels:
                        status_result = "Working"
                    else:
                        status_result = "Distracted"
                        
                    detection_history.append(status_result)
                    final_status, final_color = determine_final_status(detection_history)
                    
                    display_txt = f"{final_status}: {display_label} ({top_conf:.2f})"

                    with result_lock:
                        latest_result['box'] = (x1, y1, x2, y2)
                        latest_result['display_text'] = display_txt
                        latest_result['color'] = final_color
                        latest_result['found'] = True
                else:
                    with result_lock: latest_result['found'] = False

            elif found_person and mode_at_capture == "Break":
                with result_lock:
                    latest_result['box'] = (x1, y1, x2, y2)
                    latest_result['display_text'] = "Mode: Break (Paused)"
                    latest_result['color'] = (255, 255, 0) # Cyan/Yellowish
                    latest_result['found'] = True
            else:
                with result_lock:
                    latest_result['found'] = False

            frame_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in worker thread: {e}")

# ------------------------
# 5. MAIN LOOP
# ------------------------
# Start Thread
worker_t = threading.Thread(target=inference_worker, daemon=True)
worker_t.start()

cap = cv2.VideoCapture(0)
# Set resolusi kamera (opsional, sesuaikan kemampuan webcam)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam (index 0).")

pid = os.getpid()
process = psutil.Process(pid)

print("\nControls:")
print(" 'm' : Toggle Mode (Working/Break)")
print(" 'q' : Quit")

prev_time = time.time()
frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        frame_h, frame_w = frame.shape[:2]
        
        # Kirim frame ke thread worker setiap N frame
        if frame_count % DETECT_EVERY_N_FRAMES == 0:
            if frame_queue.empty():
                # Convert BGR (OpenCV) ke RGB (untuk Model & PIL)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                frame_data = {
                    'img': rgb_frame,
                    'size': (frame_h, frame_w),
                    'mode': current_mode
                }
                frame_queue.put(frame_data)

        # Gambar hasil deteksi terakhir (diambil dari shared variable)
        with result_lock:
            if latest_result['found'] and latest_result['box']:
                x1, y1, x2, y2 = latest_result['box']
                color = latest_result['color']
                text = latest_result['display_text']
                
                # Gambar Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Gambar Label
                draw_label(frame, text, x1, y1 - 4, color=color)

        # --- Statistik Layar ---
        cur_time = time.time()
        fps = 1.0 / (cur_time - prev_time) if (cur_time - prev_time) > 0 else 0.0
        prev_time = cur_time

        cpu_usage = process.cpu_percent(interval=None)
        ram_usage_mb = process.memory_info().rss / (1024 * 1024)

        # UI Overlay
        ui_color = (255, 255, 0)
        draw_label(frame, f"Mode: {current_mode}", 10, frame_h - 100, color=ui_color)
        draw_label(frame, f"CPU: {cpu_usage:.1f}%", 10, frame_h - 70, color=ui_color)
        draw_label(frame, f"RAM: {ram_usage_mb:.1f}MB", 10, frame_h - 40, color=ui_color)
        draw_label(frame, f"FPS: {fps:.1f}", 10, frame_h - 10, color=ui_color)

        cv2.imshow("Smart Activity Monitor", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
            break
        elif key == ord('m'):
            current_mode = "Break" if current_mode == "Working" else "Working"
            detection_history.clear() # Reset history saat ganti mode
            print(f"Mode switched to: {current_mode}")

finally:
    running = False
    cap.release()
    cv2.destroyAllWindows()
    print("Exiting...")