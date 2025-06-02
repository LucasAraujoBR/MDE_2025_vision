import os
import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

classes_map = {c: 0 for c in '0123456789'}
classes_map['+'] = 1
classes_map['='] = 2
classes_map['-'] = 3

IMAGES_PATH = 'data/images/val'
LABELS_PATH = 'data/labels/val'
DEBUG_PATH = 'data/debug'

os.makedirs(LABELS_PATH, exist_ok=True)
os.makedirs(DEBUG_PATH, exist_ok=True)

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def detect_horizontal_lines(gray_img, min_length=30, max_thickness=5):
    edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, 
                            minLineLength=min_length, maxLineGap=5)
    detected_lines = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            thickness = abs(y2 - y1)
            length = abs(x2 - x1)
            if thickness <= max_thickness and length >= min_length:
                detected_lines.append((x1, y1, x2, y2))
    return detected_lines

def detect_small_contours(gray_img, min_area=50, max_area=300):
    """Detecta contornos pequenos — potenciais operadores."""
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, x + w, y + h))
    return boxes

for filename in os.listdir(IMAGES_PATH):
    if not filename.endswith(".png"):
        continue

    image_path = os.path.join(IMAGES_PATH, filename)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro ao carregar imagem {filename}")
        continue

    h, w = img.shape[:2]

    approaches = [
        ('original', img.copy(), '--psm 11'),
        ('segunda', cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), '--psm 6'),
        ('agressiva', cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255,
                       cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2), '--psm 6'),
        ('CLAHE', apply_clahe(img.copy()), '--psm 6')
    ]

    detected = False

    for label, proc_img, psm_config in approaches:
        boxes = pytesseract.image_to_boxes(proc_img, config=psm_config)

        if not boxes.strip():
            continue

        label_file = os.path.join(LABELS_PATH, filename.replace(".png", ".txt"))
        with open(label_file, 'w') as f:
            debug_img = proc_img.copy()

            for b in boxes.strip().splitlines():
                parts = b.split()
                char = parts[0]
                x1, y1, x2, y2 = map(int, parts[1:5])
                y1_new = h - y2
                y2_new = h - y1

                cv2.rectangle(debug_img, (x1, y1_new), (x2, y2_new), (0, 255, 0), 1)
                cv2.putText(debug_img, char, (x1, y1_new - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                if char not in classes_map:
                    print(f"⚠️ Caractere ignorado: '{char}' na imagem {filename} ({label})")
                    continue

                x_center = (x1 + x2) / 2 / w
                y_center = (y1_new + y2_new) / 2 / h
                bbox_width = (x2 - x1) / w
                bbox_height = (y2_new - y1_new) / h
                class_id = classes_map[char]

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

            # Heurística extra: detectar pequenos contornos
            gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY) if len(proc_img.shape) == 3 else proc_img
            small_contours = detect_small_contours(gray)

            for x1, y1, x2, y2 in small_contours:
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 255), 1)
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                bbox_width = abs(x2 - x1) / w
                bbox_height = abs(y2 - y1) / h

                # Aqui: opcional classificar como "+"
                class_id = classes_map['+']

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

            debug_out = os.path.join(DEBUG_PATH, f"{filename.replace('.png', '')}_{label}.png")
            cv2.imwrite(debug_out, debug_img)

        detected = True
        print(f"✅ {filename}: Detectado com a abordagem {label}.")
        break

    if not detected:
        print(f"❌ {filename}: Nenhum texto detectado em nenhuma abordagem.")

print("✅ Anotações YOLO geradas com sucesso.")
