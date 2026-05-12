# EduFocus: Classroom Attention Monitoring System

**A Real-Time, Privacy-Preserving Student Attention Analysis System using Deep Learning**


---

## 📋 Overview

EduFocus is an AI-powered system that monitors student attention levels in real-time using a single classroom camera while ensuring complete privacy.

### Key Features
- Real-time attention classification (Focused / Moderately Attentive / Distracted)
- Zone-wise Attention Heatmap (Front, Middle, Back)
- Temporal Attention Curve
- Automatic Gaussian Face Blurring
- Live Flask Web Dashboard
- Automated PDF Reports

---

## 🛠️ Tech Stack
- Python 3.13
- PyTorch + MobileNetV2 + LSTM
- OpenCV
- Flask
- ReportLab

---

## 🚀 How to Run
```bash
git clone https://github.com/Aparna-rh/EduFocus.git
cd EduFocus
pip install -r requirements.txt
python app.py

Open browser → http://localhost:5000

📊 Highlights

15–18 FPS on CPU-only laptop
Strong privacy protection
Zone-wise & Temporal analysis
Aligned with SDG 4: Quality Education
