# SignCare AI: Real-Time ASL Alphabet Translator

**Built via AI-Augmented Engineering:** The computer vision pipeline, model training scripts, and backend architecture were rapidly developed and optimized using **Trae AI IDE** and **Claude**.

SignCare AI is a low-latency American Sign Language (ASL) translation engine that processes live video feeds through a custom-trained machine learning model to classify A-Z hand gestures in real time.

> **📸 [Insert a GIF or clean screenshot here showing the live webcam feed with MediaPipe landmarks and the predicted letter overlay]**

## ⚙️ The Inference Pipeline

1. **Capture & Process:** OpenCV captures the live raw video frame.
2. **Feature Extraction:** MediaPipe extracts 21 hand landmarks (63 3D spatial coordinates) and normalizes them, stripping background noise.
3. **Classification:** A lightweight Random Forest model (trained via scikit-learn) classifies the coordinate data into A-Z gestures.
4. **Delivery:** The FastAPI backend serves the translated text with near-zero latency.

## 🛠️ Tech Stack

* **Computer Vision:** OpenCV, MediaPipe
* **Machine Learning:** scikit-learn (Random Forest)
* **Backend API:** Python 3.11, FastAPI
* **Frontend:** HTML, CSS, JavaScript

## ⚡ Architectural Impact

* **Kinematic Feature Extraction:** Built a highly robust data extraction pipeline that relies entirely on 3D spatial coordinates rather than raw pixel data, ensuring the model remains lightweight and immune to lighting changes.
* **Edge-Ready Model:** Trained and hyper-tuned a Random Forest classifier to prioritize high-accuracy classification while maintaining a small enough footprint for edge device deployment.
* **Real-Time API Integration:** Wrapped the entire inference engine in an asynchronous FastAPI backend to process video frames and return results instantly.

---

## 🚀 Local Setup & Installation

**Prerequisites:** Python 3.11+ is recommended. Ensure you have a working webcam connected.

### 1. Clone the Repository

```powershell
git clone https://github.com/AK-4756/SignCare-AI.git
cd SignCare-AI

```

### 2. Configure Virtual Environment & Dependencies

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

```

### 3. Run the Inference API

```powershell
python app.py

```

*Interactive API documentation will be available at: http://localhost:8000/docs*

