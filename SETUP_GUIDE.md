# 🤟 SignCare AI - Complete Setup Guide

## Quick Start (5 minutes)

### Step 1: Set Up FastAPI Backend

```bash
# Navigate to your project directory
cd your-signcare-project

# Install Python dependencies
pip install -r requirements.txt

# Train the Kaggle model used by live recognition
python train_kaggle_model.py

# Start the FastAPI server
python app.py
```

**Expected Output:**
```
🚀 Starting SignCare AI FastAPI Server
📍 Server running at: http://localhost:8000
📚 API Documentation: http://localhost:8000/docs
```

### Step 2: Run the Frontend

**Option A: Live Server in VS Code**
1. Open `signcare-ai-backend.html` in VS Code
2. Right-click → "Open with Live Server"
3. Browser opens at `http://localhost:5500`

**Option B: Python Web Server**
```bash
# In a new terminal
python -m http.server 8001
# Visit: http://localhost:8001/signcare-ai-backend.html
```

### Step 3: Test the Application

1. ✅ Check API Status (top right corner should show "✅ API Connected")
2. ✅ Click "▶ Start Camera"
3. ✅ Make hand signs (A-Z)
4. ✅ Watch letters appear in real-time
5. ✅ AI suggests complete words
6. ✅ Build sentences

---

## 🏗️ Project Structure

```
your-signcare-project/
│
├── 📁 model/
│   ├── sign_language_model.p          # Original model trained from your CSV
│   ├── sign_language_model_kaggle.p   # Kaggle-landmark-only model
│   └── sign_language_model_combined.p # Active live model trained on combined data
│
├── 📁 Data/
│   └── alphabet_data.csv              # Original collected landmark data
│
├── 📄 combined_training_data.csv      # Merged original and Kaggle landmarks
├── 📄 train_model.py                  # Trainer for the combined dataset
├── 📄 app.py                          # FastAPI backend (NEW - provided)
├── 📄 requirements.txt                # Dependencies (NEW - provided)
│
└── 📄 signcare-ai-backend.html        # Frontend (NEW - provided)
```

---

## 📋 Detailed Setup Instructions

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**If installation fails:**

```bash
# For Windows
pip install --upgrade pip
pip install fastapi uvicorn scikit-learn opencv-python mediapipe

# For Mac/Linux
pip3 install --upgrade pip
pip3 install fastapi uvicorn scikit-learn opencv-python mediapipe
```

### 2. Train the Kaggle-Landmark-Only Model (Optional)

```bash
python train_kaggle_model.py
```

This reads the A-Z `.npy` landmarks in `archive (10)/landmarks`, converts them to wrist-relative features, adds horizontally mirrored training examples so the model can handle either hand, and saves `model/sign_language_model_kaggle.p`. It does not overwrite the active combined model.

To train the model used by live webcam inference from `combined_training_data.csv`, run:

```bash
python train_model.py
```

It uses a stratified 80/20 split, reports test accuracy, and saves `model/sign_language_model_combined.p`, which is the model loaded by the backend. The combined-data model has the additional `del`, `nothing`, and `space` classes.

### 3. Train the Movement Model for ASL J and Z

The regular model recognizes a single hand pose. J and Z are movement-based signs, so the project includes a separate sequence model. It does not replace either static-letter model.

1. Start the backend with `python app.py`, open `http://localhost:8000`, and allow camera access.
2. In **Moving-letter training (J/Z)**, select a label, click **Capture 24-frame sample**, and perform the complete sign while keeping your hand visible.
3. Capture at least 30 varied samples for each label. **One button click records one sample** made from 24 time-spaced camera frames (about 2.4 seconds). The backend finds hand landmarks from those images in sequence; brief tracking gaps are interpolated, but a sample is rejected if fewer than 18 frames contain a hand or a tracking gap lasts more than 4 frames. The J, Z, and OTHER counters show saved samples versus the 30-sample minimum; a counter is complete at `30 / 30`:
   - **J**: perform the standard ASL J movement.
   - **Z**: perform the standard ASL Z movement.
   - **OTHER**: capture static letters and unrelated hand movement, not J or Z.
4. Stop the backend with `Ctrl+C`, then train the motion model:

   ```bash
   python train_motion_model.py
   ```

5. Review the held-out classification report, restart the backend, and check `http://localhost:8000/motion-status`. The motion recognizer loads `model/motion_sequence_model.p` when the backend starts.

Collect examples with varied movement speed, camera distance, and (where possible) different people. The Kaggle `archive (10)/landmarks` data contains individual poses rather than ordered movement recordings, so it cannot replace the webcam sequence samples needed for J/Z motion training.

### 4. Compare the Own and Kaggle Static-Letter Datasets

Run the evaluation without replacing either model:

```bash
python evaluate_datasets.py
```

It reports both saved models, fresh grouped holdout models, and cross-dataset scores on original and horizontally mirrored poses. For the Kaggle model, the grouped split happens before mirror augmentation, and only training samples are augmented; this prevents synthetic mirrors of held-out poses from leaking into training. The Kaggle coordinates are converted to wrist-relative features to match the application. The datasets do not identify people, so these results do not establish accuracy on unseen signers; that requires testing recordings from people not represented in training.

### 5. Start the FastAPI Backend

```bash
python app.py
```

**Or use Uvicorn directly:**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Available Endpoints:**
- `http://localhost:8000/` - API Status
- `http://localhost:8000/docs` - Interactive API Documentation
- `http://localhost:8000/predict` - Send image for prediction
- `http://localhost:8000/suggestions/{prefix}` - Get word suggestions

### 6. Run the Frontend

**In VS Code:**
1. Install "Live Server" extension (if not installed)
2. Open `signcare-ai-backend.html`
3. Right-click → "Open with Live Server"
4. Automatically opens at `http://localhost:5500`

**Or manually:**
```bash
# Open the HTML file directly in your browser
# Click "Open File" and select signcare-ai-backend.html
```

---

## 🔧 Configuration

### Change Backend URL

If running on a different machine, edit `signcare-ai-backend.html`:

```javascript
// Find this line (around line 280):
const API_URL = 'http://localhost:8000';

// Change to your server IP:
const API_URL = 'http://192.168.1.100:8000';
// or for remote deployment:
const API_URL = 'https://your-api.com';
```

### Change Camera Settings

In `app.py`, modify the MediaPipe initialization:

```python
self.hands = self.mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,  # 0-1 (lower = more lenient)
    min_tracking_confidence=0.5
)
```

### Adjust Prediction Stability

In `signcare-ai-backend.html`, find:

```javascript
const REQUIRED_STABLE_FRAMES = 15;  // Number of frames to confirm prediction
const PREDICTION_INTERVAL = 300;     // Milliseconds between predictions
```

---

## 🚀 Deployment (Production)

### Option 1: Deploy Backend to Heroku

```bash
# Login to Heroku
heroku login

# Create app
heroku create your-signcare-app

# Add model file
git add model/sign_language_model_kaggle.p
git commit -m "Add trained model"

# Deploy
git push heroku main

# View logs
heroku logs --tail
```

### Option 2: Deploy to Railway.app

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Link project
railway link

# Deploy
railway up
```

### Option 3: Docker Deployment

**Create Dockerfile:**
```dockerfile
FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and run:**
```bash
docker build -t signcare-ai .
docker run -p 8000:8000 signcare-ai
```

### Option 4: Deploy Frontend to Netlify

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Deploy
netlify deploy --prod --files . --dir .

# Or drag and drop signcare-ai-backend.html on netlify.com
```

---

## 🐛 Troubleshooting

### "Cannot Connect to API"

**Problem:** Frontend shows "❌ API Disconnected"

**Solution:**
1. Make sure `python app.py` is running
2. Check backend is at `http://localhost:8000`
3. Try `http://localhost:8000/docs` - should show API documentation
4. Check Windows Firewall isn't blocking port 8000

```bash
# On Windows (PowerShell):
netstat -ano | findstr :8000

# On Mac/Linux:
lsof -i :8000
```

### "Camera Permission Denied"

**Problem:** Camera won't start

**Solution:**
1. Allow browser camera access when prompted
2. Check browser settings → Privacy → Camera
3. Restart browser
4. Try different browser (Chrome works best)

### "Model File Not Found"

**Problem:** `FileNotFoundError: model/sign_language_model_combined.p`

**Solution:**
```bash
# Train the active combined-data model
python train_model.py

# Verify file exists
ls model/sign_language_model_combined.p
```

### "Predictions are all wrong"

**Problem:** Predictions don't match hand signs

**Solution:**
1. Check model accuracy:
   ```bash
   # Add this to app.py to see model info
   http://localhost:8000/model-info
   ```
2. Collect more training data for difficult letters
3. Retrain model:
   ```bash
   python train.py
   ```
4. Adjust confidence threshold in `app.py`:
   ```python
   min_detection_confidence=0.7  # Lower = more sensitive
   ```

### "Slow Predictions"

**Problem:** Real-time predictions lag

**Solution:**
1. Reduce camera resolution:
   ```javascript
   width: { ideal: 640 },   // Lower = faster
   height: { ideal: 480 }
   ```
2. Increase prediction interval:
   ```javascript
   const PREDICTION_INTERVAL = 500;  // Increase from 300
   ```
3. Use GPU (requires additional setup)

---

## 📊 API Endpoints Reference

### GET `/health`
Check if API is running
```
Response: { "status": "healthy", "predictor": "ready" }
```

### POST `/predict`
Send image file for prediction
```
Body: multipart/form-data with image file
Response: {
  "letter": "A",
  "confidence": 0.95,
  "hand_detected": true
}
```

### POST `/predict-base64`
Send base64 encoded image
```
Body: { "image": "base64_string" }
Response: Same as /predict
```

### GET `/suggestions/{prefix}`
Get word suggestions
```
Example: /suggestions/HE
Response: {
  "prefix": "HE",
  "suggestions": ["HELLO", "HELP", "HEAR"],
  "count": 3
}
```

### GET `/words`
Get all available words
```
Response: {
  "total_words": 150,
  "words": ["HELLO", "HELP", ...]
}
```

### GET `/model-info`
Get model information
```
Response: {
  "status": "loaded",
  "model_type": "RandomForestClassifier",
  "classes": ["A", "B", "C", ...],
  "n_classes": 26
}
```

---

## 🎯 Next Steps

### Add More Features

1. **Save predictions to database:**
   ```python
   # In app.py
   from sqlalchemy import create_engine
   engine = create_engine('sqlite:///predictions.db')
   ```

2. **Add user authentication:**
   ```python
   from fastapi.security import HTTPBearer
   security = HTTPBearer()
   ```

3. **Real-time WebSocket:**
   ```python
   from fastapi import WebSocket
   
   @app.websocket("/ws")
   async def websocket_endpoint(websocket: WebSocket):
       await websocket.accept()
       # Stream predictions in real-time
   ```

4. **Mobile app (React Native):**
   Same API endpoints work for mobile!

---

## 📞 Support

If you encounter issues:

1. ✅ Check API is running: `http://localhost:8000/docs`
2. ✅ Check browser console for errors: F12 → Console
3. ✅ Check logs in terminal
4. ✅ Verify model file exists
5. ✅ Try restarting both backend and frontend

---

## 🎉 Success Checklist

- [ ] API running at localhost:8000
- [ ] Frontend running at localhost:5500
- [ ] API status shows "✅ API Connected"
- [ ] Camera permission allowed
- [ ] Start camera shows live video
- [ ] Letters detected and displayed
- [ ] Word suggestions appear
- [ ] Confidence score shows 60%+
- [ ] Words can be built and sentences formed
- [ ] Text-to-speech works

---

**Congratulations! SignCare AI is now fully integrated! 🚀**

Start signing and see your predictions in real-time! 🤟
