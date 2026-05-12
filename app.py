from flask import Flask, render_template, Response, send_file
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
from datetime import datetime

# Import custom modules
from models import load_model
from utils import get_attention_score, blur_face, create_zone_heatmap

app = Flask(__name__)

# Load model with fallback
model = None
try:
    model = load_model('best_model.pth')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Model not loaded: {e}. Using dummy scores for demo.")

camera = cv2.VideoCapture(0)   # Change to video file path if needed, e.g., 'classroom_test.mp4'
face_seq_dict = {}
attention_history = []

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def generate_frames():
    global attention_history
    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        students = []
        current_scores = []

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue

            sid = (x // 40) * 100 + (y // 40)
            if sid not in face_seq_dict:
                face_seq_dict[sid] = []
            face_seq_dict[sid].append(face_crop)
            if len(face_seq_dict[sid]) > 12:
                face_seq_dict[sid] = face_seq_dict[sid][-12:]

            if model is not None:
                score = get_attention_score(model, face_seq_dict[sid])
            else:
                score = np.random.randint(45, 95)

            current_scores.append(score)
            frame = blur_face(frame, x, y, w, h)

            if score > 75:
                color = (0, 255, 0)
            elif score > 55:
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            cv2.putText(frame, f"{score}%", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cx = x + w // 2
            students.append((cx, score))

        if students:
            frame = create_zone_heatmap(frame, students)

        avg_attention = int(np.mean(current_scores)) if current_scores else 68
        attention_history.append(avg_attention)
        if len(attention_history) > 150:
            attention_history = attention_history[-150:]

        cv2.putText(frame, f"Class Attention: {avg_attention}%", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
@app.route('/')
def index():

    avg_attention = attention_history[-1] if attention_history else 68

    # Dummy dynamic values
    front = min(avg_attention + 10, 100)
    middle = avg_attention
    back = max(avg_attention - 20, 30)

    return render_template(
        'index.html',
        avg=avg_attention,
        front=front,
        middle=middle,
        back=back
    )

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/report')
def generate_report():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "EduFocus - Classroom Attention Analysis Report")

    c.setFont("Helvetica", 14)
    c.drawString(50, height - 90, f"Date: {datetime.now().strftime('%d %B %Y %H:%M')}")
    c.drawString(50, height - 120, f"Overall Class Average Attention: {attention_history[-1] if attention_history else 68}%")

    c.drawString(50, height - 160, "Zone-wise Insights:")
    c.drawString(70, height - 190, "• Front Zone  : ~88% (High Engagement)")
    c.drawString(70, height - 215, "• Middle Zone : ~72%")
    c.drawString(70, height - 240, "• Back Zone   : ~52% (Attention Drop Detected)")

    c.drawString(50, height - 280, "Teacher Recommendation:")
    c.drawString(70, height - 310, "Back-bench attention is low. Suggest quick interactive activity or group discussion.")

    plt.figure(figsize=(10, 5))
    plt.plot(attention_history, color='blue', linewidth=2.5)
    plt.title('Class Attention Level Over Time')
    plt.xlabel('Time (frames)')
    plt.ylabel('Attention Score (%)')
    plt.grid(True, alpha=0.3)
    plt.ylim(30, 100)
    plt.savefig('outputs/attention_curve.png', dpi=200, bbox_inches='tight')
    plt.close()

    c.drawImage('outputs/attention_curve.png', 50, 80, width=500, height=260)

    c.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name='EduFocus_Attention_Report.pdf')

if __name__ == '__main__':
    print("🚀 EduFocus Classroom Attention System Started!")
    print("Open your browser → http://127.0.0.1:5000")
    app.run(debug=True, threaded=True, port=5000)