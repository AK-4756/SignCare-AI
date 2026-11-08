# ⚡ SignCare AI - Quick Start (5 Minutes)

## Copy-Paste Commands

### Terminal 1: Start Backend
```bash
cd your-project-folder
pip install -r requirements.txt
python app.py
```

Wait for:
```
✅ SignLanguagePredictor initialized
🚀 Starting SignCare AI FastAPI Server
📍 Server running at: http://localhost:8000
```

### Terminal 2: Start Frontend
```bash
# Option A: If you have Python
python -m http.server 8001

# Option B: Use Live Server in VS Code
# Open signcare-ai-backend.html → Right-click → Open with Live Server
```

---

## In Your Browser

1. **Backend API Docs:** `http://localhost:8000/docs`
2. **Frontend:** `http://localhost:8001/signcare-ai-backend.html`
3. Check top-right corner shows: **✅ API Connected**

---

## Test Sequence

✅ Step 1: See green "API Connected" in top-right  
✅ Step 2: Click "▶ Start Camera"  
✅ Step 3: Allow camera access  
✅ Step 4: Make hand sign (any letter A-Z)  
✅ Step 5: See letter appear in display  
✅ Step 6: Watch confidence bar fill  
✅ Step 7: Build words and sentences  

---

## Files You Got

| File | Purpose |
|------|---------|
| `app.py` | FastAPI backend with ML model |
| `requirements.txt` | Python dependencies |
| `signcare-ai-backend.html` | Frontend with real predictions |
| `SETUP_GUIDE.md` | Detailed setup instructions |
| `QUICK_START.md` | This file |

---

## Troubleshoot

### "API Disconnected"
```bash
# Make sure backend is running
python app.py

# Check at http://localhost:8000/docs
```

### "Camera won't turn on"
- Allow browser camera permission
- Try Chrome (best support)
- Restart browser

### "No predictions"
- Check hand is clearly visible
- Ensure good lighting
- Check the active combined model exists: `model/sign_language_model_combined.p`

### "Model not found"
```bash
python train_kaggle_model.py
```

---

## Key Features

✅ Real-time hand detection  
✅ Letter-by-letter recognition  
✅ Confidence scoring (0-100%)  
✅ AI word suggestions  
✅ Build complete sentences  
✅ Text-to-speech output  
✅ History saving  
✅ Multi-language support  

---

## Architecture

```
Frontend (HTML/JavaScript)
        ↓ (HTTP requests)
FastAPI Backend (Python)
        ↓ (image processing)
ML Model (scikit-learn)
        ↓ (hand detection)
MediaPipe (hand landmarks)
        ↓ (camera stream)
Your Webcam
```

---

## What Happens When You Start

1. **Backend (`app.py`):**
   - Loads trained ML model
   - Initializes MediaPipe
   - Starts FastAPI server at port 8000
   - Ready to accept predictions

2. **Frontend (`signcare-ai-backend.html`):**
   - Opens in browser
   - Connects to backend API
   - Shows "API Connected" ✅
   - Ready for camera input

3. **When you start camera:**
   - Captures video frames
   - Sends to backend every 300ms
   - Backend extracts hand landmarks
   - ML model predicts letter
   - Returns confidence score
   - Frontend displays results

4. **When prediction is stable:**
   - Same letter detected 15 times in a row
   - Letter added to word automatically
   - Suggestions updated instantly

---

## Next Steps

1. **Deploy Backend:**
   - Heroku, Railway, AWS, or Google Cloud
   - Changes API URL: `http://your-api.com`

2. **Deploy Frontend:**
   - Netlify, GitHub Pages, or Vercel
   - Drag-drop `signcare-ai-backend.html`

3. **Mobile App:**
   - Use same API endpoints
   - Build with React Native or Flutter

4. **Improve Accuracy:**
   - Collect more training data
   - Retrain model with new data
   - Adjust confidence thresholds

---

## Common Shortcuts

- **Press 1-5:** Select AI suggestion
- **Space Bar:** Add space between words
- **🔊 Button:** Speak the sentence
- **💾 Button:** Save to history
- **🗑 Button:** Clear word/sentence

---

## Performance Tips

- **Faster predictions:** Lower resolution (640x480)
- **Better accuracy:** Increase confidence threshold
- **Smoother UI:** Reduce prediction interval to 200ms
- **Lower latency:** Deploy backend nearby geographically

---

## When Everything Works

You'll see:
- ✅ Live video feed from camera
- ✅ Hand landmarks drawn
- ✅ Confidence bar filling
- ✅ Letters appearing as you sign
- ✅ Words building up
- ✅ Suggestions updating
- ✅ Complete sentences forming
- ✅ Text-to-speech output

**That's SignCare AI working perfectly!** 🎉

---

## Questions?

1. Check API docs: `http://localhost:8000/docs`
2. Check browser console: F12 → Console
3. Check terminal logs for backend errors
4. Verify the active model file exists: `model/sign_language_model_combined.p`

---

**Let's go! 🤟**
