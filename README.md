# SignCare AI: Real-Time ASL Alphabet Translator

Leveraged AI/ML Honors focus to engineer a low-latency American Sign Language (ASL) translation engine, processing live video feeds through custom-trained machine learning models to classify hand gestures in real time.

## Tech Stack
* **Computer Vision:** OpenCV, MediaPipe
* **Machine Learning:** scikit-learn (Random Forest)
* **Backend API:** Python, FastAPI

## Architectural Impact
* **Kinematic Feature Extraction:** Utilized Google's MediaPipe and OpenCV to capture and normalize 3D hand-landmark coordinates from raw webcam feeds, stripping away background noise for clean model inputs.
* **Optimized Model Training:** Trained and tuned a Random Forest classifier using scikit-learn on extracted coordinate data, prioritizing high-accuracy classification while keeping the model lightweight enough for edge deployment.
* **Real-Time API Integration:** Wrapped the inference engine in a FastAPI backend to process video frames and return translated ASL text with near-zero latency.

## Architecture & Project Structure

```text
SignCare-AI/
├── app.py                      # FastAPI backend and inference API entry point
├── requirements.txt            # Python dependencies
├── model/                      # Saved trained model artifacts
│   ├── sign_language_model.p
│   └── sign_language_model_combined.p
├── src/                        # Data preparation, training, and evaluation scripts
│   ├── clean_data.py
│   ├── collect_data.py
│   ├── motion_recognition.py
│   └── train_model.py
├── Data/                       # Hand landmark coordinate datasets
│   └── alphabet_data.csv
├── templates/                  # Frontend HTML interfaces
├── static/                     # Frontend CSS and JS assets
├── tests/                      # Automated test suite
└── docs/                       # Setup guides, results, and project notes
```

## Local Setup & Installation

The project dependencies are pinned in `requirements.txt`. Python 3.11+ is recommended.

### 1. Clone the repository
```powershell
git clone [https://github.com/AK-4756/SignCare-AI.git](https://github.com/AK-4756/SignCare-AI.git)
Set-Location SignCare-AI
```

### 2. Configure and install dependencies
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
*Once the API is running, open http://localhost:8000/docs to view the interactive API documentation.*
