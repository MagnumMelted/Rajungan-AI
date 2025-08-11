# detection.py
import cv2
import os
import sys
import easyocr
import numpy as np
import pandas as pd
from ultralytics import YOLO
from skimage.exposure import equalize_adapthist
from collections import deque

# Konstanta
LEBAR_LAYAR_LCD_CM = 7.5

# Variabel global
model = None
reader = None
detection_history = []  # Untuk simpan semua hasil
# Tambahan di global scope
panjang_buffer = deque(maxlen=10)
lebar_buffer = deque(maxlen=10)
berat_buffer = deque(maxlen=10)

last_values = {
    "panjang": None,
    "lebar": None,
    "berat": None
}

thresholds = {
    "panjang": 0.3,  # cm
    "lebar": 0.3,    # cm
    "berat": 1.0     # gram
}

# Fungsi resource path (agar PyInstaller kompatibel)
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# Inisialisasi Model YOLO & EasyOCR
def init_models():
    global model, reader
    print("INFO: Memuat model YOLO...")
    MODEL_PATH = resource_path("assets/best.pt")
    model = YOLO(MODEL_PATH)
    print("INFO: Model YOLO berhasil dimuat.")

    print("INFO: Memuat model EasyOCR...")
    try:
        OCR_MODEL_PATH = resource_path("assets/easyocr")
        reader = easyocr.Reader(['en'], gpu=True, model_storage_directory=OCR_MODEL_PATH)
        print("INFO: EasyOCR dimuat ke GPU.")
    except Exception:
        reader = easyocr.Reader(['en'], gpu=False)
        print("INFO: EasyOCR dimuat ke CPU.")


# Fungsi OCR untuk berat
def get_weight_from_screen(image, coords, ocr_reader):
    x, y, w, h = cv2.boundingRect(coords.astype(int))
    screen_img = image[y:y+h, x:x+w]
    if screen_img.size == 0:
        return None

    target_height = 64
    h_orig, w_orig, _ = screen_img.shape
    if h_orig == 0:
        return None
    ratio = target_height / h_orig
    target_width = int(w_orig * ratio)
    resized_img = cv2.resize(screen_img, (target_width, target_height))

    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    clahe_img = equalize_adapthist(gray_img, clip_limit=0.03)
    enhanced_img = (clahe_img * 255).astype("uint8")

    ocr_result = ocr_reader.readtext(enhanced_img, allowlist='0123456789.')
    if ocr_result:
        try:
            return float(ocr_result[0][1])
        except (ValueError, IndexError):
            return None
    return None


# Fungsi untuk ukur panjang/lebar
def get_size_from_contours(all_contours, all_class_ids, class_map):
    detected_labels = {class_map[int(cid)]: contour for cid, contour in zip(all_class_ids, all_contours)}
    body_label = 'carapace' if 'carapace' in detected_labels else 'ventral_view' if 'ventral_view' in detected_labels else None
    if not body_label:
        return None

    rasio_cm_per_px = None
    body_contour = detected_labels[body_label]
    x_min, y_min = body_contour.min(axis=0)
    x_max, y_max = body_contour.max(axis=0)
    lebar_body_px = x_max - x_min
    panjang_body_px = y_max - y_min

    if 'ruler' in detected_labels:
        ref_contour = detected_labels['ruler']
        ref_x_min, _ = ref_contour.min(axis=0)
        ref_x_max, _ = ref_contour.max(axis=0)
        rasio_cm_per_px = 15.0 / (ref_x_max - ref_x_min)
    elif 'weight' in detected_labels:
        ref_contour = detected_labels['weight']
        ref_x_min, _ = ref_contour.min(axis=0)
        ref_x_max, _ = ref_contour.max(axis=0)
        rasio_cm_per_px = LEBAR_LAYAR_LCD_CM / (ref_x_max - ref_x_min)
    else:
        return None

    lebar_cm = lebar_body_px * rasio_cm_per_px
    panjang_cm = panjang_body_px * rasio_cm_per_px
    return {"lebar_cm": lebar_cm, "panjang_cm": panjang_cm, "tipe_badan": body_label}


# Fungsi proses frame
def process_frame(frame):
    global detection_history, panjang_buffer, lebar_buffer, berat_buffer, last_values

    results = model(frame, device=0, stream=True, verbose=False)

    berat_rajungan = None
    ukuran_rajungan = None

    for r in results:
        all_contours = r.masks.xy if r.masks is not None else []
        all_boxes = r.boxes.data.cpu().numpy() if r.boxes is not None else []
        all_class_ids = all_boxes[:, 5] if len(all_boxes) > 0 else []

        class_names = [model.names[int(cid)] for cid in all_class_ids]

        if 'weight' in class_names:
            lcd_index = class_names.index('weight')
            berat_rajungan = get_weight_from_screen(frame, all_contours[lcd_index], reader)

        ukuran_rajungan = get_size_from_contours(all_contours, all_class_ids, model.names)

        for i, contour in enumerate(all_contours):
            class_id = int(all_class_ids[i])
            class_name = model.names[class_id]
            color = (0, 255, 0) if class_name in ['carapace', 'ventral_view'] else \
                    (255, 0, 0) if class_name in ['weight', 'layar_lcd'] else \
                    (0, 0, 255) if class_name == 'ruler' else (255, 255, 255)
            cv2.polylines(frame, [contour.astype(int)], isClosed=True, color=color, thickness=2)

    y_pos = 30
    if berat_rajungan is not None:
        cv2.putText(frame, f"Berat: {berat_rajungan:.1f} gram", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        y_pos += 40
    if ukuran_rajungan is not None:
        cv2.putText(frame, f"Lebar: {ukuran_rajungan['lebar_cm']:.2f} cm", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        y_pos += 40
        cv2.putText(frame, f"Panjang: {ukuran_rajungan['panjang_cm']:.2f} cm", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Masukkan ke buffer hanya jika berat_rajungan valid (bukan None)
    if berat_rajungan is not None and ukuran_rajungan is not None:
        # Ubah maxlen buffer berat jadi 5 untuk responsivitas lebih cepat
        if berat_buffer.maxlen != 5:
            berat_buffer = deque(berat_buffer, maxlen=5)

        berat_buffer.append(berat_rajungan)
        panjang_buffer.append(ukuran_rajungan['panjang_cm'])
        lebar_buffer.append(ukuran_rajungan['lebar_cm'])

        avg_berat = sum(berat_buffer) / len(berat_buffer)
        avg_panjang = sum(panjang_buffer) / len(panjang_buffer)
        avg_lebar = sum(lebar_buffer) / len(lebar_buffer)

        def is_significant_change(new, old, thresh):
            if old is None:
                return True
            return abs(new - old) >= thresh

        if is_significant_change(avg_berat, last_values["berat"], thresholds["berat"]):
            last_values["berat"] = avg_berat
        if is_significant_change(avg_panjang, last_values["panjang"], thresholds["panjang"]):
            last_values["panjang"] = avg_panjang
        if is_significant_change(avg_lebar, last_values["lebar"], thresholds["lebar"]):
            last_values["lebar"] = avg_lebar

        if None not in last_values.values():
            detection_history.append({
                "Berat (gram)": round(last_values["berat"], 1),
                "Lebar (cm)": round(last_values["lebar"], 2),
                "Panjang (cm)": round(last_values["panjang"], 2)
            })

    # Jika berat_rajungan None, kembalikan last_values["berat"] agar stabil
    return frame, last_values["berat"], {
        "lebar_cm": last_values["lebar"],
        "panjang_cm": last_values["panjang"],
        "tipe_badan": ukuran_rajungan['tipe_badan'] if ukuran_rajungan else None
    }


# Fungsi export ke Excel
def export_to_excel(filepath="hasil_deteksi.xlsx"):
    global detection_history  # pastikan bisa akses variabel global
    if not detection_history:
        return None
    last_entry = detection_history[-1]
    df = pd.DataFrame([last_entry])
    df.to_excel(filepath, index=False)

    # Reset detection_history supaya kosong setelah export
    detection_history = []

    return filepath
