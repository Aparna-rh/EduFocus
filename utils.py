import cv2
import numpy as np
import torch
from PIL import Image
import io

# Load OpenCV face detector (comes with opencv-python)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_attention_score(model, face_crop_seq):
    if len(face_crop_seq) < 8:  # smaller sequence for speed
        return 65
    tensor_seq = []
    for crop in face_crop_seq[-8:]:
        if crop.size == 0: continue
        crop = cv2.resize(crop, (224, 224))
        crop = torch.tensor(crop).permute(2, 0, 1).float() / 255.0
        tensor_seq.append(crop)
    if not tensor_seq:
        return 65
    tensor_seq = torch.stack(tensor_seq).unsqueeze(0)
    with torch.no_grad():
        pred = model(tensor_seq)
        score = torch.softmax(pred, dim=1)[0][0].item() * 100
    return int(score)

def blur_face(frame, x, y, w, h):
    face = frame[y:y+h, x:x+w]
    if face.size == 0: return frame
    face = cv2.GaussianBlur(face, (51, 51), 30)
    frame[y:y+h, x:x+w] = face
    return frame

def create_zone_heatmap(frame, student_data):
    h, w = frame.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.uint8)
    for cx, score in student_data:
        intensity = int(score * 2.55)
        cv2.circle(heatmap, (cx, h//2), 70, intensity, -1)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 0.75, heatmap_colored, 0.25, 0)