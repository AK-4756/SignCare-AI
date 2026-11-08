"""
SignCare AI - FastAPI Backend
Real-time Sign Language Recognition with ML Model Integration
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
import base64
from io import BytesIO
from PIL import Image
import logging

# =====================================================
# Setup Logging
# =====================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================
# Initialize FastAPI
# =====================================================
app = FastAPI(
    title="SignCare AI API",
    description="Real-time Sign Language Translator",
    version="1.0.0"
)

# =====================================================
# CORS Configuration
# =====================================================
# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# Load Model and Initialize MediaPipe
# =====================================================

# =====================================================
# Landmark Validation & Temporal Stability Configuration
# =====================================================

MIN_HAND_WIDTH = 0.05
MIN_HAND_HEIGHT = 0.05
MIN_HAND_SPAN = 0.08
MAX_HAND_SIZE = 0.98

MIN_ASPECT_RATIO = 0.15
MAX_ASPECT_RATIO = 5.0

MIN_LANDMARK_SPREAD = 0.012
MIN_PALM_SIZE = 0.025
MIN_FINGER_REACH = 0.035

MAX_HAND_JUMP = 0.45
REQUIRED_VALID_FRAMES = 2
MAX_MISSING_FRAMES = 2

MIN_DETECTION_CONFIDENCE = 0.45
MIN_TRACKING_CONFIDENCE = 0.45


class SignLanguagePredictor:
    """Handles sign language prediction using trained ML model with multi-layer validation"""
    
    def __init__(self):
        """Initialize MediaPipe, load trained model, and setup temporal tracking"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.model = None
        self.load_model()
        
        # Temporal & Spatial Tracking State
        self.consecutive_valid_frames = 0
        self.missing_frames_count = 0
        self.is_hand_confirmed = False
        self.last_hand_center = None
        
        logger.info("✅ SignLanguagePredictor initialized with Landmark Validation & Temporal Filtering")
    
    def load_model(self):
        """Load the trained sklearn model"""
        model_path = os.path.join('model', 'sign_language_model.p')
        
        if not os.path.exists(model_path):
            logger.error(f"❌ Model not found at {model_path}")
            logger.error("Please run train.py first to create the model")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                model_dict = pickle.load(f)
                if isinstance(model_dict, dict) and 'model' in model_dict:
                    self.model = model_dict['model']
                else:
                    self.model = model_dict
            logger.info(f"✅ Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise

    @staticmethod
    def validate_hand_landmarks(landmarks):
        """
        Evaluate whether the 21 detected landmarks form a plausible human hand.
        """
        if not landmarks or len(landmarks) != 21:
            return False, f"Expected 21 landmarks, got {len(landmarks) if landmarks else 0}"

        xs, ys, zs = [], [], []

        for i, lm in enumerate(landmarks):
            x = lm.x if hasattr(lm, 'x') else lm.get('x', None)
            y = lm.y if hasattr(lm, 'y') else lm.get('y', None)
            z = lm.z if hasattr(lm, 'z') else lm.get('z', None)

            if x is None or y is None or z is None:
                return False, f"Missing coordinate in landmark {i}"

            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                return False, f"Non-finite coordinate in landmark {i}"

            if not (-0.20 <= x <= 1.20 and -0.20 <= y <= 1.20):
                return False, f"Landmark {i} coordinate out of reasonable bounds"

            xs.append(x)
            ys.append(y)
            zs.append(z)

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        w = max_x - min_x
        h = max_y - min_y
        span = np.hypot(w, h)

        if w < MIN_HAND_WIDTH or h < MIN_HAND_HEIGHT or span < MIN_HAND_SPAN:
            return False, "Hand bounding box too small"

        if w > MAX_HAND_SIZE and h > MAX_HAND_SIZE:
            return False, "Hand bounding box oversized"

        aspect = w / max(h, 1e-6)
        if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
            return False, "Invalid aspect ratio"

        std_x = float(np.std(xs))
        std_y = float(np.std(ys))
        if std_x < MIN_LANDMARK_SPREAD and std_y < MIN_LANDMARK_SPREAD:
            return False, "Landmarks collapsed"

        wrist_x, wrist_y = xs[0], ys[0]
        middle_mcp_x, middle_mcp_y = xs[9], ys[9]
        index_mcp_x, index_mcp_y = xs[5], ys[5]

        palm_dist = np.hypot(middle_mcp_x - wrist_x, middle_mcp_y - wrist_y)
        index_dist = np.hypot(index_mcp_x - wrist_x, index_mcp_y - wrist_y)

        if max(palm_dist, index_dist) < MIN_PALM_SIZE:
            return False, "Palm structure too small"

        fingertip_indices = [4, 8, 12, 16, 20]
        max_reach = max(np.hypot(xs[t] - wrist_x, ys[t] - wrist_y) for t in fingertip_indices)
        if max_reach < MIN_FINGER_REACH:
            return False, "Finger reach too small"

        return True, "Valid hand"
    
    def extract_landmarks(self, frame):
        """
        Extract hand landmarks from frame using MediaPipe and validate structure.
        """
        target_width = 640
        h, w = frame.shape[:2]
        scale = target_width / w
        target_height = int(h * scale)
        resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None, False, None, None, None
        
        hand_landmarks = results.multi_hand_landmarks[0]

        is_valid, reason = self.validate_hand_landmarks(hand_landmarks.landmark)
        if not is_valid:
            return None, False, None, None, None
        
        landmarks_data = []
        visual_landmarks = []
        
        wrist_x = hand_landmarks.landmark[0].x
        wrist_y = hand_landmarks.landmark[0].y
        wrist_z = hand_landmarks.landmark[0].z
        
        xs, ys = [], []
        for i in range(len(hand_landmarks.landmark)):
            lm = hand_landmarks.landmark[i]
            landmarks_data.extend([
                lm.x - wrist_x,
                lm.y - wrist_y,
                lm.z - wrist_z
            ])
            visual_landmarks.append({
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z)
            })
            xs.append(lm.x)
            ys.append(lm.y)
        
        hand_center = (float(np.mean(xs)), float(np.mean(ys)))
        return np.asarray(landmarks_data, dtype=np.float32), True, results, visual_landmarks, hand_center
    
    def predict(self, frame):
        """
        Predict sign language letter from frame with temporal and geometric validation.
        """
        (
            landmarks_data,
            candidate_valid,
            results,
            visual_landmarks,
            hand_center
        ) = self.extract_landmarks(frame)
        
        if not candidate_valid or hand_center is None:
            self.missing_frames_count += 1
            if self.missing_frames_count >= MAX_MISSING_FRAMES:
                self.consecutive_valid_frames = 0
                self.is_hand_confirmed = False
                self.last_hand_center = None

            return {
                'letter': None,
                'confidence': 0.0,
                'hand_detected': False,
                'landmarks_data': None,
                'visual_landmarks': None
            }
        
        center_x, center_y = hand_center
        if self.last_hand_center is not None and self.missing_frames_count == 0:
            jump = np.hypot(center_x - self.last_hand_center[0], center_y - self.last_hand_center[1])
            if jump > MAX_HAND_JUMP:
                self.consecutive_valid_frames = 1
                self.is_hand_confirmed = False
                self.last_hand_center = (center_x, center_y)
                self.missing_frames_count = 0
                if REQUIRED_VALID_FRAMES > 1:
                    return {
                        'letter': None,
                        'confidence': 0.0,
                        'hand_detected': False,
                        'landmarks_data': None,
                        'visual_landmarks': None
                    }
            else:
                self.consecutive_valid_frames += 1
                self.missing_frames_count = 0
                self.last_hand_center = (center_x, center_y)
        else:
            self.consecutive_valid_frames += 1
            self.missing_frames_count = 0
            self.last_hand_center = (center_x, center_y)

        if self.consecutive_valid_frames >= REQUIRED_VALID_FRAMES:
            self.is_hand_confirmed = True

        if not self.is_hand_confirmed:
            return {
                'letter': None,
                'confidence': 0.0,
                'hand_detected': False,
                'landmarks_data': None,
                'visual_landmarks': None
            }

        try:
            prediction_proba = self.model.predict_proba([landmarks_data])[0]
            confidence = float(np.max(prediction_proba))
            predicted_index = int(np.argmax(prediction_proba))
            predicted_letter = str(self.model.classes_[predicted_index])
            
            return {
                'letter': predicted_letter,
                'confidence': confidence,
                'hand_detected': True,
                'landmarks_data': landmarks_data.tolist(),
                'visual_landmarks': visual_landmarks
            }
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'letter': None,
                'confidence': 0.0,
                'hand_detected': False,
                'landmarks_data': None,
                'visual_landmarks': None
            }
    
    def draw_landmarks(self, frame, results):
        """
        Draw hand landmarks on frame
        
        Args:
            frame: Input image
            results: MediaPipe results object
            
        Returns:
            frame with landmarks drawn
        """
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(
                        color=(255, 255, 255),
                        thickness=2,
                        circle_radius=2
                    ),
                    self.mp_draw.DrawingSpec(
                        color=(230, 160, 70),
                        thickness=2
                    )
                )
        return frame


# =====================================================
# Initialize Predictor (Global)
# =====================================================
try:
    predictor = SignLanguagePredictor()
    logger.info("✅ Predictor initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize predictor: {e}")
    predictor = None

# =====================================================
# Word Dictionary
# =====================================================
WORDS = [
    # Common words
    "HELLO", "HELP", "THANK", "PLEASE", "YES", "NO",
    "GOOD", "BAD", "LOVE", "HATE", "LIKE", "WATER",
    "FOOD", "SLEEP", "WORK", "PLAY", "FRIEND", "FAMILY",
    "DOCTOR", "HOSPITAL", "PAIN", "EMERGENCY", "FEVER",
    "COLD", "HAPPY", "SAD", "TIRED", "SICK", "WELL",
    "MORNING", "NIGHT", "TODAY", "TOMORROW", "YESTERDAY",
    "HOME", "SCHOOL", "BATHROOM", "KITCHEN", "BEDROOM",
    # Medical words
    "MEDICINE", "PILL", "NURSE", "PATIENT", "HOSPITAL",
    "EMERGENCY", "CALL", "HELP", "PAIN", "HURT",
    "FEVER", "COLD", "COUGH", "SICK", "WELL",
    "DIZZY", "NAUSEA", "HEADACHE", "TIRED", "REST"
]

# =====================================================
# API Routes
# =====================================================

@app.get("/")
async def root():
    """Root endpoint - API status"""
    return {
        "status": "✅ SignCare AI API is running",
        "version": "1.0.0",
        "predictor_status": "active" if predictor else "inactive"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "predictor": "ready" if predictor and predictor.model else "not_ready"
    }

@app.post("/predict")
async def predict_sign(file: UploadFile = File(...)):
    """
    Predict sign language letter from image
    
    Expected: Image file (JPEG, PNG)
    Returns: {
        'letter': predicted letter,
        'confidence': confidence score,
        'hand_detected': bool
    }
    """
    if not predictor or not predictor.model:
        return JSONResponse(
            status_code=500,
            content={"error": "Predictor not initialized"}
        )
    
    try:
        # Read uploaded image
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        # Convert to numpy array
        frame = cv2.cvtColor(
            np.array(image),
            cv2.COLOR_RGB2BGR
        )
        
        # Make prediction
        result = predictor.predict(frame)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error in predict: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )

@app.post("/predict-base64")
async def predict_sign_base64(data: dict):
    """
    Predict sign language from base64 encoded image
    
    Expected JSON:
    {
        'image': base64_encoded_image
    }
    
    Returns: Prediction result with letter and confidence
    """
    if not predictor or not predictor.model:
        return JSONResponse(
            status_code=500,
            content={"error": "Predictor not initialized"}
        )
    
    try:
        # Decode base64 image
        image_data = data.get('image')
        if not image_data:
            return JSONResponse(
                status_code=400,
                content={"error": "No image data provided"}
            )
        
        # Remove data URI prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to BGR for OpenCV
        frame = cv2.cvtColor(
            np.array(image),
            cv2.COLOR_RGB2BGR
        )
        
        # Make prediction
        result = predictor.predict(frame)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error in predict_base64: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )

@app.get("/suggestions/{prefix}")
async def get_word_suggestions(prefix: str):
    """
    Get word suggestions based on prefix
    
    Example: /suggestions/HE returns ["HELLO", "HELP", "HEAR"]
    """
    prefix = prefix.upper().strip()
    
    if not prefix:
        return {
            "prefix": prefix,
            "suggestions": []
        }
    
    # Find matching words
    matches = [word for word in WORDS if word.startswith(prefix)]
    
    # Return top 5 suggestions
    suggestions = matches[:5]
    
    return {
        "prefix": prefix,
        "suggestions": suggestions,
        "count": len(suggestions)
    }

@app.get("/words")
async def get_all_words():
    """Get all available words in dictionary"""
    return {
        "total_words": len(WORDS),
        "words": sorted(list(set(WORDS)))
    }

@app.post("/batch-predict")
async def batch_predict(data: dict):
    """
    Process multiple frames for batch prediction
    
    Expected JSON:
    {
        'frames': [base64_image1, base64_image2, ...]
    }
    """
    if not predictor or not predictor.model:
        return JSONResponse(
            status_code=500,
            content={"error": "Predictor not initialized"}
        )
    
    try:
        frames_data = data.get('frames', [])
        results = []
        
        for frame_b64 in frames_data:
            try:
                # Decode image
                if ',' in frame_b64:
                    frame_b64 = frame_b64.split(',')[1]
                
                image_bytes = base64.b64decode(frame_b64)
                image = Image.open(BytesIO(image_bytes))
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Predict
                result = predictor.predict(frame)
                results.append(result)
            
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                results.append({"error": str(e)})
        
        return {
            "total_frames": len(frames_data),
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Error in batch_predict: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if not predictor or not predictor.model:
        return {
            "status": "not_loaded",
            "message": "Model not initialized"
        }
    
    try:
        return {
            "status": "loaded",
            "model_type": type(predictor.model).__name__,
            "classes": list(predictor.model.classes_),
            "n_classes": len(predictor.model.classes_),
            "n_features": predictor.model.n_features_in_ if hasattr(predictor.model, 'n_features_in_') else "unknown"
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

# =====================================================
# Error Handlers
# =====================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )

# =====================================================
# Startup Event
# =====================================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("🚀 SignCare AI API Starting...")
    
    if predictor and predictor.model:
        logger.info(f"✅ Model loaded with {len(predictor.model.classes_)} classes")
        logger.info(f"✅ Dictionary loaded with {len(WORDS)} words")
    else:
        logger.warning("⚠️ Warning: Model not properly initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("🛑 SignCare AI API Shutting down...")

# =====================================================
# Run FastAPI Server (for development)
# =====================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting SignCare AI FastAPI Server")
    logger.info("📍 Server running at: http://localhost:8000")
    logger.info("📚 API Documentation: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
