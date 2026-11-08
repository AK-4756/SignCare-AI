"""
SignCare AI - FastAPI Backend
Real-time Sign Language Recognition with ML Model Integration
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
import base64
import binascii
import logging
import json
import math
import urllib.request
import urllib.parse
import uuid

from io import BytesIO
from PIL import Image, ImageOps
from src.motion_recognition import (
    FEATURE_COUNT,
    MIN_SAMPLES_PER_LABEL,
    MOTION_LABELS,
    SEQUENCE_LENGTH,
    interpolate_missing_frames,
    normalize_sequence,
)


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
# Serve static images and other website assets
STATIC_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "static"
)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)

# =====================================================
# CORS Configuration
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# Project Paths
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "sign_language_model_combined_splitfirst.p")
MOTION_DATA_DIR = os.path.join(BASE_DIR, "Data", "motion_sequences")
MOTION_MODEL_PATH = os.path.join(MODEL_DIR, "motion_sequence_model.p")


def load_motion_model():
    if not os.path.exists(MOTION_MODEL_PATH):
        return None

    with open(MOTION_MODEL_PATH, "rb") as model_file:
        model_data = pickle.load(model_file)

    if (
        not isinstance(model_data, dict)
        or model_data.get("sequence_length") != SEQUENCE_LENGTH
        or model_data.get("feature_count") != FEATURE_COUNT
        or model_data.get("labels") != list(MOTION_LABELS)
        or "model" not in model_data
    ):
        raise ValueError(f"Invalid motion model artifact: {MOTION_MODEL_PATH}")

    logger.info("✅ Motion sequence model loaded from %s", MOTION_MODEL_PATH)
    return model_data["model"]


motion_model = load_motion_model()

logger.info(f"📁 Project directory: {BASE_DIR}")
logger.info(f"📁 Model path: {MODEL_PATH}")


# =====================================================
# Landmark Validation & Temporal Stability Configuration
# =====================================================

# 1. Bounding Box Dimensions (Normalized 0.0 - 1.0)
# Minimum hand bounding box width and height to reject degenerate micro-detections/false palm points
MIN_HAND_WIDTH = 0.05
MIN_HAND_HEIGHT = 0.05
# Minimum bounding box diagonal span sqrt(w^2 + h^2)
MIN_HAND_SPAN = 0.08
# Maximum bounding box size to reject unrealistic full-frame distortions
MAX_HAND_SIZE = 0.98

# 2. Aspect Ratio (Width / Height)
# Human hands in reasonable camera view stay within [0.15, 5.0]
MIN_ASPECT_RATIO = 0.15
MAX_ASPECT_RATIO = 5.0

# 3. Landmark Spread (Coordinate dispersion)
# Standard deviation of landmark coordinates to ensure points aren't collapsed to a single pixel
MIN_LANDMARK_SPREAD = 0.012

# 4. Anatomical Structure (Palm & Fingertip Reach)
# Minimum distance from wrist (0) to middle finger MCP knuckle (9) or index MCP (5)
MIN_PALM_SIZE = 0.025
# Minimum distance from wrist (0) to furthest fingertip (4, 8, 12, 16, 20)
MIN_FINGER_REACH = 0.035

# 5. Landmark Movement & Spatial Consistency
# Maximum allowed jump distance of hand center between consecutive frames (normalized 0.0 - 1.0)
MAX_HAND_JUMP = 0.45

# 6. Temporal Confirmation Threshold
# Number of consecutive valid frames required before confirming a hand detection
REQUIRED_VALID_FRAMES = 2
# Number of missing/invalid frames before clearing confirmed hand state
MAX_MISSING_FRAMES = 2

# 7. MediaPipe Detection Confidence
# Balanced threshold: avoids false palm hallucination on clothing/faces while keeping real hand detection seamless
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.45


# =====================================================
# Sign Language Predictor
# =====================================================

class SignLanguagePredictor:
    """Handles sign language prediction using trained ML model with multi-layer validation."""

    def __init__(self):
        """Initialize MediaPipe, load trained model, and setup temporal tracking."""

        # -------------------------------------------------
        # Initialize MediaPipe Hands
        # -------------------------------------------------
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )

        self.mp_draw = mp.solutions.drawing_utils

        # -------------------------------------------------
        # Load trained model
        # -------------------------------------------------
        self.model = None
        self.model_source = "Unknown"
        self.load_model()

        # -------------------------------------------------
        # Temporal & Spatial Tracking State
        # -------------------------------------------------
        self.consecutive_valid_frames = 0
        self.missing_frames_count = 0
        self.is_hand_confirmed = False
        self.last_hand_center = None

        logger.info("✅ SignLanguagePredictor initialized with Landmark Validation & Temporal Filtering")

    # =================================================
    # Load Model
    # =================================================

    def load_model(self):
        """Load the trained sklearn model."""

        if not os.path.exists(MODEL_PATH):
            logger.error(f"❌ Model not found at {MODEL_PATH}")
            logger.error("Please run src/train_model.py first.")
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        try:
            with open(MODEL_PATH, "rb") as f:
                model_dict = pickle.load(f)

            if isinstance(model_dict, dict) and "model" in model_dict:
                self.model = model_dict["model"]
                self.model_source = model_dict.get("source", "Unknown")
            else:
                self.model = model_dict

            logger.info(
                "✅ %s model loaded successfully from %s",
                self.model_source,
                MODEL_PATH,
            )

        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise

    # =================================================
    # Preprocess Image
    # =================================================

    def preprocess_frame(self, frame):
        """Prepare image before MediaPipe processing."""

        if frame is None or frame.size == 0:
            raise ValueError("Invalid or empty image")

        original_height, original_width = frame.shape[:2]

        target_width = 640
        scale = target_width / original_width
        target_height = int(original_height * scale)

        resized = cv2.resize(
            frame,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA
        )

        return resized

    # =================================================
    # Landmark Geometric & Anatomical Validation
    # =================================================

    @staticmethod
    def validate_hand_landmarks(landmarks):
        """
        Evaluate whether the 21 detected landmarks form a plausible human hand.
        Rejects false-positive MediaPipe detections on face, clothing, and background.

        Returns:
            tuple: (is_valid: bool, reason: str)
        """
        if not landmarks or len(landmarks) != 21:
            return False, f"Expected 21 landmarks, got {len(landmarks) if landmarks else 0}"

        xs = []
        ys = []
        zs = []

        for i, lm in enumerate(landmarks):
            x = lm.x if hasattr(lm, 'x') else lm.get('x', None)
            y = lm.y if hasattr(lm, 'y') else lm.get('y', None)
            z = lm.z if hasattr(lm, 'z') else lm.get('z', None)

            if x is None or y is None or z is None:
                return False, f"Missing coordinate in landmark {i}"

            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                return False, f"Non-finite coordinate in landmark {i}"

            # Ensure coordinates are within reasonable image margins
            if not (-0.20 <= x <= 1.20 and -0.20 <= y <= 1.20):
                return False, f"Landmark {i} coordinate out of reasonable image bounds ({x:.2f}, {y:.2f})"

            xs.append(x)
            ys.append(y)
            zs.append(z)

        # 1. Bounding Box & Span Check
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        w = max_x - min_x
        h = max_y - min_y
        span = math.hypot(w, h)

        if w < MIN_HAND_WIDTH or h < MIN_HAND_HEIGHT or span < MIN_HAND_SPAN:
            return False, f"Hand bounding box too small (w={w:.3f}, h={h:.3f}, span={span:.3f})"

        if w > MAX_HAND_SIZE and h > MAX_HAND_SIZE:
            return False, f"Hand bounding box unrealistically oversized (w={w:.3f}, h={h:.3f})"

        # 2. Aspect Ratio Check (reject degenerate lines)
        aspect = w / max(h, 1e-6)
        if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
            return False, f"Invalid hand aspect ratio ({aspect:.2f})"

        # 3. Landmark Spread / Dispersion (reject collapsed points)
        std_x = float(np.std(xs))
        std_y = float(np.std(ys))
        if std_x < MIN_LANDMARK_SPREAD and std_y < MIN_LANDMARK_SPREAD:
            return False, f"Landmarks collapsed (std_x={std_x:.4f}, std_y={std_y:.4f})"

        # 4. Anatomical Structure (Palm & Fingertip Reach)
        wrist_x, wrist_y = xs[0], ys[0]
        middle_mcp_x, middle_mcp_y = xs[9], ys[9]
        index_mcp_x, index_mcp_y = xs[5], ys[5]

        palm_dist = math.hypot(middle_mcp_x - wrist_x, middle_mcp_y - wrist_y)
        index_dist = math.hypot(index_mcp_x - wrist_x, index_mcp_y - wrist_y)

        if max(palm_dist, index_dist) < MIN_PALM_SIZE:
            return False, f"Wrist-to-MCP knuckle distance too small ({max(palm_dist, index_dist):.3f})"

        fingertip_indices = [4, 8, 12, 16, 20]
        max_reach = max(math.hypot(xs[t] - wrist_x, ys[t] - wrist_y) for t in fingertip_indices)
        if max_reach < MIN_FINGER_REACH:
            return False, f"Wrist-to-fingertip reach too small ({max_reach:.3f})"

        return True, "Valid hand"

    # =================================================
    # Extract Hand Landmarks
    # =================================================

    def extract_landmarks(self, frame):
        """
        Extract 21 hand landmarks using MediaPipe and validate structure.

        Returns:
            tuple:
                landmarks_data      -> 63 wrist-relative ML features (or None)
                hand_detected       -> bool
                results_object      -> MediaPipe results
                visual_landmarks    -> 21 absolute normalized points for UI overlay
                hand_center         -> (center_x, center_y)
        """

        if frame is None or frame.size == 0:
            return None, False, None, None, None

        processed_frame = self.preprocess_frame(frame)
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return None, False, None, None, None

        hand_landmarks = results.multi_hand_landmarks[0]

        # -------------------------------------------------
        # Validate Geometric Hand Plausibility
        # -------------------------------------------------
        is_valid, reason = self.validate_hand_landmarks(hand_landmarks.landmark)
        if not is_valid:
            logger.info(f"🚫 Landmark validation rejected false detection: {reason}")
            return None, False, None, None, None

        # -------------------------------------------------
        # Extract Features & Visual Points
        # -------------------------------------------------
        landmarks_data = []
        visual_landmarks = []

        wrist_x = hand_landmarks.landmark[0].x
        wrist_y = hand_landmarks.landmark[0].y
        wrist_z = hand_landmarks.landmark[0].z

        xs = []
        ys = []

        for lm in hand_landmarks.landmark:
            # 63 wrist-relative features for Random Forest
            landmarks_data.extend([
                lm.x - wrist_x,
                lm.y - wrist_y,
                lm.z - wrist_z
            ])

            # 21 normalized points for frontend UI overlay
            visual_landmarks.append({
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z)
            })

            xs.append(lm.x)
            ys.append(lm.y)

        landmarks_data = np.asarray(landmarks_data, dtype=np.float32)
        hand_center = (float(np.mean(xs)), float(np.mean(ys)))

        return landmarks_data, True, results, visual_landmarks, hand_center

    # =================================================
    # Prediction with Temporal Stability & Movement Check
    # =================================================

    def predict(self, frame):
        """
        Predict sign language letter from image with multi-layer false positive rejection.

        Returns:
            dict:
                letter
                confidence
                hand_detected
                landmarks_data      -> 63 ML features
                visual_landmarks    -> 21 UI overlay points
        """

        (
            landmarks_data,
            candidate_valid,
            results,
            visual_landmarks,
            hand_center
        ) = self.extract_landmarks(frame)

        # -------------------------------------------------
        # Case A: No valid hand detected in frame
        # -------------------------------------------------
        if not candidate_valid or hand_center is None:
            self.missing_frames_count += 1
            if self.missing_frames_count >= MAX_MISSING_FRAMES:
                self.consecutive_valid_frames = 0
                self.is_hand_confirmed = False
                self.last_hand_center = None

            return {
                "letter": None,
                "confidence": 0.0,
                "hand_detected": False,
                "landmarks_data": None,
                "visual_landmarks": None
            }

        # -------------------------------------------------
        # Case B: Candidate hand detected -> Validate Movement Jump
        # -------------------------------------------------
        center_x, center_y = hand_center

        if self.last_hand_center is not None and self.missing_frames_count == 0:
            jump = math.hypot(center_x - self.last_hand_center[0], center_y - self.last_hand_center[1])
            if jump > MAX_HAND_JUMP:
                logger.warning(
                    f"⚠️ Suspicious hand jump detected: {jump:.2f} > {MAX_HAND_JUMP}. "
                    "Resetting temporal confirmation at new location."
                )
                self.consecutive_valid_frames = 1
                self.is_hand_confirmed = False
                self.last_hand_center = (center_x, center_y)
                self.missing_frames_count = 0

                if REQUIRED_VALID_FRAMES > 1:
                    return {
                        "letter": None,
                        "confidence": 0.0,
                        "hand_detected": False,
                        "landmarks_data": None,
                        "visual_landmarks": None
                    }
            else:
                self.consecutive_valid_frames += 1
                self.missing_frames_count = 0
                self.last_hand_center = (center_x, center_y)
        else:
            self.consecutive_valid_frames += 1
            self.missing_frames_count = 0
            self.last_hand_center = (center_x, center_y)

        # -------------------------------------------------
        # Temporal Confirmation Check
        # -------------------------------------------------
        if self.consecutive_valid_frames >= REQUIRED_VALID_FRAMES:
            self.is_hand_confirmed = True

        if not self.is_hand_confirmed:
            # Candidate hand is valid but awaiting temporal confirmation
            return {
                "letter": None,
                "confidence": 0.0,
                "hand_detected": False,
                "landmarks_data": None,
                "visual_landmarks": None
            }

        # -------------------------------------------------
        # Validate Feature Count
        # -------------------------------------------------
        if len(landmarks_data) != 63:
            logger.error(
                f"❌ Invalid feature count: {len(landmarks_data)}. Expected 63."
            )
            return {
                "letter": None,
                "confidence": 0.0,
                "hand_detected": False,
                "landmarks_data": None,
                "visual_landmarks": None
            }

        # -------------------------------------------------
        # Random Forest Model Prediction
        # -------------------------------------------------
        try:
            prediction_proba = self.model.predict_proba([landmarks_data])[0]
            confidence = float(np.max(prediction_proba))
            predicted_index = int(np.argmax(prediction_proba))
            predicted_letter = str(self.model.classes_[predicted_index])

            logger.info(
                f"🎯 Validated Prediction: {predicted_letter} ({confidence:.2%})"
            )

            return {
                "letter": predicted_letter,
                "confidence": confidence,
                "hand_detected": True,
                "landmarks_data": landmarks_data.tolist(),
                "visual_landmarks": visual_landmarks
            }

        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return {
                "letter": None,
                "confidence": 0.0,
                "hand_detected": False,
                "landmarks_data": None,
                "visual_landmarks": None
            }

    # =================================================
    # Draw Landmarks
    # =================================================

    def draw_landmarks(self, frame, results):
        """Draw hand landmarks on image."""
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        return frame


# =====================================================
# Initialize Predictor
# =====================================================

try:

    predictor = SignLanguagePredictor()

    logger.info(
        "✅ Predictor initialized successfully"
    )

except Exception as e:

    logger.error(
        f"❌ Failed to initialize predictor: {e}"
    )

    predictor = None


# =====================================================
# Word Dictionary (Multilingual: English, Hindi, Gujarati)
# =====================================================

MULTILINGUAL_WORDS = {
    # A
    "APPLE": {"en": "APPLE", "hi": "सेब", "gu": "સફરજન"},
    "ABOUT": {"en": "ABOUT", "hi": "के बारे में", "gu": "વિશે"},
    "ABOVE": {"en": "ABOVE", "hi": "ऊपर", "gu": "ઉપર"},
    "AFTER": {"en": "AFTER", "hi": "बाद में", "gu": "પછી"},
    "AGAIN": {"en": "AGAIN", "hi": "फिर से", "gu": "ફરીથી"},
    "ALWAYS": {"en": "ALWAYS", "hi": "हमेशा", "gu": "હંમેશા"},
    "AM": {"en": "AM", "hi": "हूँ", "gu": "છું"},
    "AND": {"en": "AND", "hi": "और", "gu": "અને"},
    "ANIMAL": {"en": "ANIMAL", "hi": "जानवर", "gu": "પ્રાણી"},
    "ANSWER": {"en": "ANSWER", "hi": "उत्तर", "gu": "જવાબ"},
    "ANY": {"en": "ANY", "hi": "कोई भी", "gu": "કોઈપણ"},
    "ARE": {"en": "ARE", "hi": "हैं", "gu": "છે"},

    # B
    "BALL": {"en": "BALL", "hi": "गेंद", "gu": "દડો"},
    "BABY": {"en": "BABY", "hi": "बच्चा", "gu": "બાળક"},
    "BACK": {"en": "BACK", "hi": "पीछे", "gu": "પાછળ"},
    "BAD": {"en": "BAD", "hi": "बुरा", "gu": "ખરાબ"},
    "BANK": {"en": "BANK", "hi": "बैंक", "gu": "બેંક"},
    "BEAUTIFUL": {"en": "BEAUTIFUL", "hi": "सुंदर", "gu": "સુંદર"},
    "BECAUSE": {"en": "BECAUSE", "hi": "क्योंकि", "gu": "કારણ કે"},
    "BED": {"en": "BED", "hi": "बिस्तर", "gu": "પલંગ"},
    "BEFORE": {"en": "BEFORE", "hi": "पहले", "gu": "પહેલાં"},
    "BEST": {"en": "BEST", "hi": "सर्वश्रेष्ठ", "gu": "શ્રેષ્ઠ"},
    "BETTER": {"en": "BETTER", "hi": "बेहतर", "gu": "વધુ સારું"},
    "BOOK": {"en": "BOOK", "hi": "किताब", "gu": "પુસ્તક"},
    "BROTHER": {"en": "BROTHER", "hi": "भाई", "gu": "ભાઈ"},
    "BUILD": {"en": "BUILD", "hi": "बनाना", "gu": "બનાવવું"},
    "BUS": {"en": "BUS", "hi": "बस", "gu": "બસ"},
    "BUY": {"en": "BUY", "hi": "खरीदना", "gu": "ખરીદવું"},

    # C
    "CAN": {"en": "CAN", "hi": "सकना", "gu": "શકવું"},
    "CAR": {"en": "CAR", "hi": "कार", "gu": "ગાડી"},
    "CAT": {"en": "CAT", "hi": "बिल्ली", "gu": "બિલાડી"},
    "COME": {"en": "COME", "hi": "आना", "gu": "આવવું"},
    "COMPUTER": {"en": "COMPUTER", "hi": "कंप्यूटर", "gu": "કમ્પ્યુટર"},
    "COOL": {"en": "COOL", "hi": "शांत", "gu": "શાંત"},
    "COLD": {"en": "COLD", "hi": "ठंडा", "gu": "ઠંડુ"},
    "COLLEGE": {"en": "COLLEGE", "hi": "कॉलेज", "gu": "કોલેજ"},
    "CLASS": {"en": "CLASS", "hi": "कक्षा", "gu": "વર્ગ"},
    "CLEAN": {"en": "CLEAN", "hi": "साफ", "gu": "સ્વચ્છ"},
    "CLOSE": {"en": "CLOSE", "hi": "बंद", "gu": "બંધ"},

    # D
    "DAY": {"en": "DAY", "hi": "दिन", "gu": "દિવસ"},
    "DOG": {"en": "DOG", "hi": "कुत्ता", "gu": "કૂતરો"},
    "DOOR": {"en": "DOOR", "hi": "दरवाजा", "gu": "દરવાજો"},
    "DOCTOR": {"en": "DOCTOR", "hi": "डॉक्टर", "gu": "ડોક્ટર"},
    "DAD": {"en": "DAD", "hi": "पिताजी", "gu": "પપ્પા"},
    "DANCE": {"en": "DANCE", "hi": "नाच", "gu": "નૃત્ય"},
    "DRINK": {"en": "DRINK", "hi": "पीना", "gu": "પીવું"},
    "DRIVE": {"en": "DRIVE", "hi": "चलाना", "gu": "ચલાવવું"},

    # E
    "EAT": {"en": "EAT", "hi": "खाना", "gu": "ખાવું"},
    "EVERY": {"en": "EVERY", "hi": "हर", "gu": "દરેક"},
    "EASY": {"en": "EASY", "hi": "आसान", "gu": "સરળ"},
    "ENJOY": {"en": "ENJOY", "hi": "आनंद लेना", "gu": "આનંદ માણવો"},
    "END": {"en": "END", "hi": "अंत", "gu": "અંત"},
    "EXAMPLE": {"en": "EXAMPLE", "hi": "उदाहरण", "gu": "ઉદાહરણ"},

    # F
    "FAMILY": {"en": "FAMILY", "hi": "परिवार", "gu": "પરિવાર"},
    "FATHER": {"en": "FATHER", "hi": "पिता", "gu": "પિતા"},
    "FRIEND": {"en": "FRIEND", "hi": "दोस्त", "gu": "મિત્ર"},
    "FOOD": {"en": "FOOD", "hi": "खाना", "gu": "ખોરાક"},
    "FINE": {"en": "FINE", "hi": "ठीक", "gu": "સારું"},
    "FIRST": {"en": "FIRST", "hi": "पहला", "gu": "પ્રથમ"},
    "FOR": {"en": "FOR", "hi": "के लिए", "gu": "માટે"},
    "FROM": {"en": "FROM", "hi": "से", "gu": "તરફથી"},

    # G
    "GOOD": {"en": "GOOD", "hi": "अच्छा", "gu": "સારું"},
    "GREAT": {"en": "GREAT", "hi": "महान", "gu": "મહાન"},
    "GO": {"en": "GO", "hi": "जाना", "gu": "જવું"},
    "GIVE": {"en": "GIVE", "hi": "देना", "gu": "આપવું"},
    "GET": {"en": "GET", "hi": "पाना", "gu": "મેળવવું"},
    "GOODBYE": {"en": "GOODBYE", "hi": "अलविदा", "gu": "આવજો"},

    # H
    "HELLO": {"en": "HELLO", "hi": "नमस्ते", "gu": "નમસ્તે"},
    "HELP": {"en": "HELP", "hi": "मदद", "gu": "મદદ"},
    "HOME": {"en": "HOME", "hi": "घर", "gu": "ઘર"},
    "HOPE": {"en": "HOPE", "hi": "आशा", "gu": "આશા"},
    "HOW": {"en": "HOW", "hi": "कैसे", "gu": "કેવી રીતે"},
    "HAPPY": {"en": "HAPPY", "hi": "खुश", "gu": "ખુશ"},
    "HAVE": {"en": "HAVE", "hi": "पास होना", "gu": "પાસે હોવું"},
    "HE": {"en": "HE", "hi": "वह", "gu": "તે"},
    "HER": {"en": "HER", "hi": "उसकी", "gu": "તેણીની"},
    "HERE": {"en": "HERE", "hi": "यहाँ", "gu": "અહીં"},

    # I
    "I": {"en": "I", "hi": "मैं", "gu": "હું"},
    "IS": {"en": "IS", "hi": "है", "gu": "છે"},
    "IT": {"en": "IT", "hi": "यह", "gu": "તે"},
    "IN": {"en": "IN", "hi": "में", "gu": "માં"},
    "IMPORTANT": {"en": "IMPORTANT", "hi": "महत्वपूर्ण", "gu": "મહત્વપૂર્ણ"},
    "INTERESTING": {"en": "INTERESTING", "hi": "रोचक", "gu": "રસપ્રદ"},

    # J
    "JOB": {"en": "JOB", "hi": "नौकरी", "gu": "નોકરી"},
    "JUST": {"en": "JUST", "hi": "बस", "gu": "બસ"},

    # K
    "KEEP": {"en": "KEEP", "hi": "रखना", "gu": "રાખવું"},
    "KNOW": {"en": "KNOW", "hi": "जानना", "gu": "જાણવું"},
    "KIND": {"en": "KIND", "hi": "दयालु", "gu": "દયાળુ"},

    # L
    "LOVE": {"en": "LOVE", "hi": "प्यार", "gu": "પ્રેમ"},
    "LIKE": {"en": "LIKE", "hi": "पसंद", "gu": "પસંદ"},
    "LIFE": {"en": "LIFE", "hi": "जीवन", "gu": "જીવન"},
    "LITTLE": {"en": "LITTLE", "hi": "छोटा", "gu": "નાનું"},
    "LOOK": {"en": "LOOK", "hi": "देखना", "gu": "જોવું"},
    "LEARN": {"en": "LEARN", "hi": "सीखना", "gu": "શીખવું"},
    "LET": {"en": "LET", "hi": "देना", "gu": "દેવું"},

    # M
    "MOTHER": {"en": "MOTHER", "hi": "माँ", "gu": "માતા"},
    "MY": {"en": "MY", "hi": "मेरा", "gu": "મારું"},
    "ME": {"en": "ME", "hi": "मुझे", "gu": "મને"},
    "MORE": {"en": "MORE", "hi": "अधिक", "gu": "વધુ"},
    "MORNING": {"en": "MORNING", "hi": "सुबह", "gu": "સવાર"},
    "MAKE": {"en": "MAKE", "hi": "बनाना", "gu": "બનાવવું"},
    "MAN": {"en": "MAN", "hi": "पुरुष", "gu": "પુરુષ"},
    "MANY": {"en": "MANY", "hi": "अनेक", "gu": "ઘણા"},

    # N
    "NAME": {"en": "NAME", "hi": "नाम", "gu": "નામ"},
    "NICE": {"en": "NICE", "hi": "अच्छा", "gu": "સરસ"},
    "NO": {"en": "NO", "hi": "नहीं", "gu": "ના"},
    "NOT": {"en": "NOT", "hi": "नहीं", "gu": "નથી"},
    "NOW": {"en": "NOW", "hi": "अब", "gu": "હવે"},
    "NEW": {"en": "NEW", "hi": "नया", "gu": "નવું"},
    "NEED": {"en": "NEED", "hi": "जरूरत", "gu": "જરૂર"},

    # O
    "OF": {"en": "OF", "hi": "का", "gu": "નું"},
    "OK": {"en": "OK", "hi": "ठीक है", "gu": "બરાબર"},
    "OLD": {"en": "OLD", "hi": "पुराना", "gu": "જૂનું"},
    "ON": {"en": "ON", "hi": "पर", "gu": "પર"},
    "ONE": {"en": "ONE", "hi": "एक", "gu": "એક"},
    "ONLY": {"en": "ONLY", "hi": "केवल", "gu": "માત્ર"},
    "OPEN": {"en": "OPEN", "hi": "खोलना", "gu": "ખોલવું"},

    # P
    "PLEASE": {"en": "PLEASE", "hi": "कृपया", "gu": "કૃપા કરીને"},
    "PEOPLE": {"en": "PEOPLE", "hi": "लोग", "gu": "લોકો"},
    "PHONE": {"en": "PHONE", "hi": "फोन", "gu": "ફોન"},
    "PLACE": {"en": "PLACE", "hi": "स्थान", "gu": "જગ્યા"},
    "PLAY": {"en": "PLAY", "hi": "खेलना", "gu": "રમવું"},
    "PUT": {"en": "PUT", "hi": "रखना", "gu": "મૂકવું"},
    "PROBLEM": {"en": "PROBLEM", "hi": "समस्या", "gu": "સમસ્યા"},

    # Q
    "QUESTION": {"en": "QUESTION", "hi": "सवाल", "gu": "પ્રશ્ન"},
    "QUICK": {"en": "QUICK", "hi": "तेज", "gu": "ઝડપી"},

    # R
    "READ": {"en": "READ", "hi": "पढ़ना", "gu": "વાંચવું"},
    "READY": {"en": "READY", "hi": "तैयार", "gu": "તૈયાર"},
    "RIGHT": {"en": "RIGHT", "hi": "सही", "gu": "સાચું"},
    "RUN": {"en": "RUN", "hi": "दौड़ना", "gu": "દોડવું"},
    "REALLY": {"en": "REALLY", "hi": "वास्तव में", "gu": "ખરેખર"},
    "REMEMBER": {"en": "REMEMBER", "hi": "याद रखना", "gu": "યાદ રાખવું"},

    # S
    "SCHOOL": {"en": "SCHOOL", "hi": "विद्यालय", "gu": "શાળા"},
    "SEE": {"en": "SEE", "hi": "देखना", "gu": "જોવું"},
    "SHE": {"en": "SHE", "hi": "वह", "gu": "તેણી"},
    "SIGN": {"en": "SIGN", "hi": "संकेत", "gu": "સંકેત"},
    "SIGNAL": {"en": "SIGNAL", "hi": "इशारा", "gu": "ઈશારો"},
    "SIMPLE": {"en": "SIMPLE", "hi": "सरल", "gu": "સરળ"},
    "SORRY": {"en": "SORRY", "hi": "माफ़ कीजिए", "gu": "માફ કરશો"},
    "START": {"en": "START", "hi": "शुरू करना", "gu": "શરૂ કરવું"},
    "STOP": {"en": "STOP", "hi": "रुकना", "gu": "ઊભા રહેવું"},
    "STUDENT": {"en": "STUDENT", "hi": "छात्र", "gu": "વિદ્યાર્થી"},
    "SPEAK": {"en": "SPEAK", "hi": "बोलना", "gu": "બોલવું"},
    "SOME": {"en": "SOME", "hi": "कुछ", "gu": "થોડું"},

    # T
    "THANK": {"en": "THANK", "hi": "धन्यवाद", "gu": "આભાર"},
    "THANKS": {"en": "THANKS", "hi": "धन्यवाद", "gu": "આભાર"},
    "THE": {"en": "THE", "hi": "वह", "gu": "તે"},
    "THIS": {"en": "THIS", "hi": "यह", "gu": "આ"},
    "TODAY": {"en": "TODAY", "hi": "आज", "gu": "આજે"},
    "TIME": {"en": "TIME", "hi": "समय", "gu": "સમય"},
    "TO": {"en": "TO", "hi": "को", "gu": "તરફ"},
    "TAKE": {"en": "TAKE", "hi": "लेना", "gu": "લેવું"},
    "TEACH": {"en": "TEACH", "hi": "सिखाना", "gu": "શીખવવું"},
    "THINK": {"en": "THINK", "hi": "सोचना", "gu": "વિચારવું"},

    # U
    "UNDER": {"en": "UNDER", "hi": "नीचे", "gu": "નીચે"},
    "USE": {"en": "USE", "hi": "उपयोग", "gu": "ઉપયોગ"},
    "UP": {"en": "UP", "hi": "ऊपर", "gu": "ઉપર"},
    "UNDERSTAND": {"en": "UNDERSTAND", "hi": "समझना", "gu": "સમજવું"},

    # V
    "VERY": {"en": "VERY", "hi": "बहुत", "gu": "ખૂબ"},
    "VOICE": {"en": "VOICE", "hi": "आवाज", "gu": "અવાજ"},
    "VISIT": {"en": "VISIT", "hi": "मुलाकात", "gu": "મુલાકાત"},

    # W
    "WATER": {"en": "WATER", "hi": "पानी", "gu": "પાણી"},
    "WELCOME": {"en": "WELCOME", "hi": "स्वागत", "gu": "સ્વાગત"},
    "WHAT": {"en": "WHAT", "hi": "क्या", "gu": "શું"},
    "WHERE": {"en": "WHERE", "hi": "कहाँ", "gu": "ક્યાં"},
    "WORD": {"en": "WORD", "hi": "शब्द", "gu": "શબ્દ"},
    "WORK": {"en": "WORK", "hi": "काम", "gu": "કામ"},
    "WANT": {"en": "WANT", "hi": "चाहना", "gu": "ઇચ્છવું"},
    "WE": {"en": "WE", "hi": "हम", "gu": "અમે"},
    "WELL": {"en": "WELL", "hi": "अच्छा", "gu": "સારું"},
    "WHO": {"en": "WHO", "hi": "कौन", "gu": "કોણ"},
    "WITH": {"en": "WITH", "hi": "साथ", "gu": "સાથે"},
    "WRITE": {"en": "WRITE", "hi": "लिखना", "gu": "લખવું"},

    # Y
    "YES": {"en": "YES", "hi": "हाँ", "gu": "હા"},
    "YOU": {"en": "YOU", "hi": "आप", "gu": "તમે"},
    "YOUR": {"en": "YOUR", "hi": "आपका", "gu": "તમારું"},
    "YOUNG": {"en": "YOUNG", "hi": "युवा", "gu": "યુવાન"},

    # Z
    "ZERO": {"en": "ZERO", "hi": "शून्य", "gu": "શૂન્ય"}
}

# Preserve ordered English word list for ASL prefix matching
WORDS = list(MULTILINGUAL_WORDS.keys())


# =====================================================
# Root Endpoint
# =====================================================

@app.get("/")
async def root():
    return FileResponse(
        os.path.join(BASE_DIR, "templates", "signcare-ai.html")
    )


# =====================================================
# Health Check
# =====================================================

@app.get("/health")
async def health_check():

    return {
        "status": "healthy",
        "predictor": (
            "ready"
            if predictor and predictor.model
            else "not_ready"
        )
    }


@app.get("/motion-status")
async def motion_status():
    sample_counts = {}
    for label in MOTION_LABELS:
        label_dir = os.path.join(MOTION_DATA_DIR, label)
        sample_counts[label] = (
            len([name for name in os.listdir(label_dir) if name.endswith(".npy")])
            if os.path.isdir(label_dir)
            else 0
        )

    return {
        "ready": motion_model is not None,
        "sequence_length": SEQUENCE_LENGTH,
        "feature_count": FEATURE_COUNT,
        "minimum_samples_per_label": MIN_SAMPLES_PER_LABEL,
        "labels": list(MOTION_LABELS),
        "sample_counts": sample_counts,
    }


@app.post("/motion-samples")
async def save_motion_sample(data: dict):
    label = str(data.get("label", "")).strip().upper()
    if label not in MOTION_LABELS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Label must be one of {', '.join(MOTION_LABELS)}"},
        )

    try:
        frames = np.asarray(data.get("frames"), dtype=np.float32)
    except (TypeError, ValueError) as error:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid motion frames: {error}"},
        )

    if frames.shape != (SEQUENCE_LENGTH, FEATURE_COUNT):
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    f"Expected {SEQUENCE_LENGTH} frames with {FEATURE_COUNT} features "
                    f"each; received shape {frames.shape}"
                )
            },
        )
    if not np.isfinite(frames).all():
        return JSONResponse(
            status_code=400,
            content={"error": "Motion frames contain non-finite values"},
        )
    try:
        normalize_sequence(frames)
    except ValueError as error:
        return JSONResponse(status_code=400, content={"error": str(error)})

    label_dir = os.path.join(MOTION_DATA_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    sample_path = os.path.join(label_dir, f"{uuid.uuid4().hex}.npy")
    with open(sample_path, "wb") as sample_file:
        np.save(sample_file, frames, allow_pickle=False)

    sample_count = len(
        [name for name in os.listdir(label_dir) if name.endswith(".npy")]
    )
    return {"saved": True, "label": label, "sample_count": sample_count}


@app.post("/motion-samples-from-video")
async def save_motion_sample_from_video(data: dict):
    if predictor is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Hand-landmark detector is unavailable; check backend startup logs"},
        )

    label = str(data.get("label", "")).strip().upper()
    if label not in MOTION_LABELS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Label must be one of {', '.join(MOTION_LABELS)}"},
        )

    images = data.get("images")
    if not isinstance(images, list) or len(images) != SEQUENCE_LENGTH:
        return JSONResponse(
            status_code=400,
            content={"error": f"Expected exactly {SEQUENCE_LENGTH} captured camera frames"},
        )

    sequence = []
    for frame_index, image_data in enumerate(images):
        if not isinstance(image_data, str):
            return JSONResponse(
                status_code=400,
                content={"error": f"Camera frame {frame_index + 1} is invalid"},
            )
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(image_data, validate=True)
            if not image_bytes or len(image_bytes) > 2_000_000:
                raise ValueError("empty or oversized image")
            image = Image.open(BytesIO(image_bytes))
            if image.width > 1920 or image.height > 1080:
                raise ValueError("camera frame dimensions exceed 1920x1080")
            image = ImageOps.exif_transpose(image).convert("RGB")
            frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        except (binascii.Error, OSError, ValueError, Image.DecompressionBombError) as error:
            return JSONResponse(
                status_code=400,
                content={"error": f"Could not read camera frame {frame_index + 1}: {error}"},
            )

        landmarks_data, hand_detected, _, _, _ = predictor.extract_landmarks(frame)
        sequence.append(landmarks_data if hand_detected else None)

    try:
        frames, valid_frame_count = interpolate_missing_frames(sequence)
        normalize_sequence(frames)
    except ValueError as error:
        return JSONResponse(status_code=400, content={"error": str(error)})

    label_dir = os.path.join(MOTION_DATA_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    sample_path = os.path.join(label_dir, f"{uuid.uuid4().hex}.npy")
    with open(sample_path, "wb") as sample_file:
        np.save(sample_file, frames, allow_pickle=False)

    sample_count = len(
        [name for name in os.listdir(label_dir) if name.endswith(".npy")]
    )
    return {
        "saved": True,
        "label": label,
        "sample_count": sample_count,
        "valid_frames": valid_frame_count,
        "total_frames": SEQUENCE_LENGTH,
    }


@app.post("/predict-motion")
async def predict_motion(data: dict):
    if motion_model is None:
        return {"ready": False, "label": None, "confidence": 0.0}

    try:
        frames = data.get("frames")
        if not isinstance(frames, list) or not 2 <= len(frames) <= 64:
            return JSONResponse(
                status_code=400,
                content={"error": "Motion input must contain between 2 and 64 frames"},
            )
        sequence = normalize_sequence(frames).reshape(1, -1)
    except (TypeError, ValueError) as error:
        return JSONResponse(status_code=400, content={"error": str(error)})

    probabilities = motion_model.predict_proba(sequence)[0]
    predicted_index = int(np.argmax(probabilities))
    return {
        "ready": True,
        "label": str(motion_model.classes_[predicted_index]),
        "confidence": float(probabilities[predicted_index]),
    }


# =====================================================
# Predict Sign
# =====================================================

@app.post("/predict")
async def predict_sign(
    file: UploadFile = File(...)
):
    """
    Predict sign language letter from image.

    Expected:
        JPEG / PNG image

    Returns:
        letter
        confidence
        hand_detected
        landmarks_data
    """

    if not predictor or not predictor.model:

        return JSONResponse(
            status_code=500,
            content={
                "error": "Predictor not initialized"
            }
        )

    try:

        # -------------------------------------------------
        # Check file type
        # -------------------------------------------------

        if file.content_type and not file.content_type.startswith(
            "image/"
        ):

            return JSONResponse(
                status_code=400,
                content={
                    "error": "Please upload an image file."
                }
            )

        # -------------------------------------------------
        # Read uploaded image
        # -------------------------------------------------

        contents = await file.read()

        if not contents:

            return JSONResponse(
                status_code=400,
                content={
                    "error": "Uploaded file is empty."
                }
            )

        # -------------------------------------------------
        # Open image with PIL
        # -------------------------------------------------

        image = Image.open(
            BytesIO(contents)
        )

        # Handle image orientation
        image = ImageOps.exif_transpose(image)

        # Convert to RGB
        image = image.convert("RGB")

        # -------------------------------------------------
        # Convert PIL -> OpenCV BGR
        # -------------------------------------------------

        frame = cv2.cvtColor(
            np.array(image),
            cv2.COLOR_RGB2BGR
        )

        logger.info(
            f"📷 Received image: "
            f"{image.width}x{image.height}"
        )

        # -------------------------------------------------
        # Predict
        # -------------------------------------------------

        result = predictor.predict(frame)

        return JSONResponse(
            content=result
        )

    except Exception as e:

        logger.exception(
            "❌ Error in /predict"
        )

        return JSONResponse(
            status_code=400,
            content={
                "error": str(e)
            }
        )


# =====================================================
# Predict Sign - Base64
# =====================================================

@app.post("/predict-base64")
async def predict_sign_base64(
    data: dict
):
    """
    Predict sign language from base64 image.

    Expected:
        {
            "image": "base64_encoded_image"
        }
    """

    if not predictor or not predictor.model:

        return JSONResponse(
            status_code=500,
            content={
                "error": "Predictor not initialized"
            }
        )

    try:

        image_data = data.get("image")

        if not image_data:

            return JSONResponse(
                status_code=400,
                content={
                    "error": "No image data provided"
                }
            )

        # -------------------------------------------------
        # Remove data URI prefix
        # -------------------------------------------------

        if "," in image_data:

            image_data = image_data.split(
                ",",
                1
            )[1]

        # -------------------------------------------------
        # Decode Base64
        # -------------------------------------------------

        image_bytes = base64.b64decode(
            image_data
        )

        image = Image.open(
            BytesIO(image_bytes)
        )

        image = ImageOps.exif_transpose(
            image
        )

        image = image.convert("RGB")

        # -------------------------------------------------
        # Convert to OpenCV
        # -------------------------------------------------

        frame = cv2.cvtColor(
            np.array(image),
            cv2.COLOR_RGB2BGR
        )

        # -------------------------------------------------
        # Predict
        # -------------------------------------------------

        result = predictor.predict(frame)

        return JSONResponse(
            content=result
        )

    except Exception as e:

        logger.exception(
            "❌ Error in /predict-base64"
        )

        return JSONResponse(
            status_code=400,
            content={
                "error": str(e)
            }
        )


# =====================================================
# Word Suggestions (Multilingual)
# =====================================================

@app.get("/suggestions/{prefix}")
async def get_suggestions(prefix: str, language: str = "en"):
    """
    Return up to 3 words that start with the complete current prefix.
    Supports English ('en'), Hindi ('hi'), and Gujarati ('gu').
    Defaults to English when no language parameter is provided.
    """
    prefix = prefix.upper().strip()
    target_lang = (language or "en").lower().strip()
    if target_lang not in ("en", "hi", "gu"):
        target_lang = "en"

    if not prefix:
        return {
            "prefix": prefix,
            "language": target_lang,
            "suggestions": [],
            "count": 0
        }

    # Keep the dictionary order so the first 3 matching words
    # are the first 3 words in WORDS.
    matches = [
        word for word in WORDS
        if word.startswith(prefix)
    ]

    matched_words = matches[:3]

    suggestions = []
    for word in matched_words:
        trans_info = MULTILINGUAL_WORDS.get(word, {})
        display_text = trans_info.get(target_lang) or word
        suggestions.append({
            "english": word,
            "display": display_text
        })

    return {
        "prefix": prefix,
        "language": target_lang,
        "suggestions": suggestions,
        "count": len(suggestions)
    }


@app.get("/words")
async def get_words(language: str = "en"):
    """
    Return default dictionary words when prefix is empty.
    Localized according to requested language ('en', 'hi', 'gu').
    """
    target_lang = (language or "en").lower().strip()
    if target_lang not in ("en", "hi", "gu"):
        target_lang = "en"

    default_words = ["HELLO", "HELP", "HOPE", "THANK", "LOVE", "FRIEND"]
    localized_words = []
    for word in default_words:
        trans_info = MULTILINGUAL_WORDS.get(word, {})
        display_text = trans_info.get(target_lang) or word
        localized_words.append({
            "english": word,
            "display": display_text
        })

    return {
        "language": target_lang,
        "words": localized_words,
        "count": len(localized_words)
    }


# =====================================================
# Multilingual Translation Support (Indian & Global)
# =====================================================

LANGUAGE_CONFIG = {
    "en": {"name": "English", "code": "en", "voice": "en-IN", "flag": "🇺🇸/🇮🇳"},
    "hi": {"name": "Hindi (हिन्दी)", "code": "hi", "voice": "hi-IN", "flag": "🇮🇳"},
    "gu": {"name": "Gujarati (ગુજરાતી)", "code": "gu", "voice": "gu-IN", "flag": "🇮🇳"},
    "mr": {"name": "Marathi (मराठी)", "code": "mr", "voice": "mr-IN", "flag": "🇮🇳"},
    "bn": {"name": "Bengali (বাংলা)", "code": "bn", "voice": "bn-IN", "flag": "🇮🇳"},
    "ta": {"name": "Tamil (தமிழ்)", "code": "ta", "voice": "ta-IN", "flag": "🇮🇳"},
    "te": {"name": "Telugu (తెలుగు)", "code": "te", "voice": "te-IN", "flag": "🇮🇳"},
    "kn": {"name": "Kannada (ಕನ್ನಡ)", "code": "kn", "voice": "kn-IN", "flag": "🇮🇳"},
    "ml": {"name": "Malayalam (മലയാളം)", "code": "ml", "voice": "ml-IN", "flag": "🇮🇳"},
    "pa": {"name": "Punjabi (ਪੰਜਾਬੀ)", "code": "pa", "voice": "pa-IN", "flag": "🇮🇳"},
    "ur": {"name": "Urdu (اردو)", "code": "ur", "voice": "ur-IN", "flag": "🇮🇳"},
    "or": {"name": "Odia (ଓଡ଼ିଆ)", "code": "or", "voice": "or-IN", "flag": "🇮🇳"},
    "as": {"name": "Assamese (অসমীয়া)", "code": "as", "voice": "as-IN", "flag": "🇮🇳"},
    "sa": {"name": "Sanskrit (संस्कृतम्)", "code": "sa", "voice": "sa-IN", "flag": "🇮🇳"},
    "es": {"name": "Spanish (Español)", "code": "es", "voice": "es-ES", "flag": "🇪🇸"},
    "fr": {"name": "French (Français)", "code": "fr", "voice": "fr-FR", "flag": "🇫🇷"}
}

TRANSLATION_DICTIONARY = {
    "HELLO": {"hi": "नमस्ते", "gu": "નમસ્તે", "mr": "नमस्कार", "bn": "নমস্কার", "ta": "வணக்கம்", "te": "నమస్కారం", "kn": "ನಮಸ್ಕಾರ", "ml": "നമസ്കാരം", "pa": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ", "ur": "سلام", "or": "ନମସ୍କାର", "as": "নমস্কাৰ", "sa": "नमस्ते", "es": "Hola", "fr": "Bonjour"},
    "HELP": {"hi": "मदद", "gu": "મદદ", "mr": "मदत", "bn": "সাহায্য", "ta": "உதவி", "te": "సహాయం", "kn": "ಸಹಾಯ", "ml": "സഹായം", "pa": "ਮਦਦ", "ur": "مدد", "or": "ସାହାଯ୍ୟ", "as": "সহায়", "sa": "सहायता", "es": "Ayuda", "fr": "Aide"},
    "THANK YOU": {"hi": "धन्यवाद", "gu": "આભાર", "mr": "धन्यवाद", "bn": "ধন্যবাদ", "ta": "நன்றி", "te": "ధన్యవాదాలు", "kn": "ಧನ್ಯವಾದಗಳು", "ml": "നന്ദി", "pa": "ਧੰਨਵਾਦ", "ur": "شکریہ", "or": "ଧନ୍ୟବାଦ", "as": "ধন্যবাদ", "sa": "धन्यवादः", "es": "Gracias", "fr": "Merci"},
    "THANKS": {"hi": "धन्यवाद", "gu": "આભાર", "mr": "धन्यवाद", "bn": "ধন্যবাদ", "ta": "நன்றி", "te": "ధన్యవాదాలు", "kn": "ಧನ್ಯವಾದಗಳು", "ml": "നന്ദി", "pa": "ਧੰਨਵਾਦ", "ur": "شکریہ", "or": "ଧନ୍ୟବାଦ", "as": "ধন্যবাদ", "sa": "धन्यवादः", "es": "Gracias", "fr": "Merci"},
    "PLEASE": {"hi": "कृपया", "gu": "કૃપા કરીને", "mr": "कृपया", "bn": "দয়া করে", "ta": "தயவுசெய்து", "te": "దయచేసి", "kn": "ದಯವಿಟ್ಟು", "ml": "ദയവായി", "pa": "ਕਿਰਪਾ ਕਰਕੇ", "ur": "براہ کرم", "or": "ଦୟାକରି", "as": "অনুগ্ৰহ কৰি", "sa": "कृपया", "es": "Por favor", "fr": "S'il vous plaît"},
    "YES": {"hi": "हाँ", "gu": "હા", "mr": "हो", "bn": "হ্যাঁ", "ta": "ஆம்", "te": "అవును", "kn": "ಹೌದು", "ml": "അതെ", "pa": "ਹਾਂ", "ur": "ہاں", "or": "ହଁ", "as": "হয়", "sa": "आम्", "es": "Sí", "fr": "Oui"},
    "NO": {"hi": "नहीं", "gu": "ના", "mr": "नाही", "bn": "না", "ta": "இல்லை", "te": "కాదు", "kn": "ಇಲ್ಲ", "ml": "ಇಲ್ಲ", "pa": "ਨਹੀਂ", "ur": "نہیں", "or": "ନା", "as": "নহয়", "sa": "न", "es": "No", "fr": "Non"},
    "WATER": {"hi": "पानी", "gu": "પાણી", "mr": "પાણી", "bn": "জল", "ta": "தண்ணீர்", "te": "నీరు", "kn": "ನೀರು", "ml": "വെള്ളം", "pa": "ਪਾਣੀ", "ur": "پانی", "or": "ପାଣି", "as": "পানী", "sa": "जलम्", "es": "Agua", "fr": "Eau"},
    "FOOD": {"hi": "खाना", "gu": "ખોરાક", "mr": "अन्न", "bn": "খাবার", "ta": "உணவு", "te": "ఆహారం", "kn": "ಆಹಾರ", "ml": "ഭക്ഷണം", "pa": "ਖਾਣਾ", "ur": "کھانا", "or": "ଖାଦ୍ୟ", "as": "খাদ্য", "sa": "भोजनम्", "es": "Comida", "fr": "Nourriture"},
    "DOCTOR": {"hi": "डॉक्टर", "gu": "ડોક્ટર", "mr": "डॉक्टर", "bn": "ডাক্তার", "ta": "மருத்துவர்", "te": "వైద్యుడు", "kn": "ವೈದ್ಯರು", "ml": "ഡോക്ടർ", "pa": "ਡਾਕਟਰ", "ur": "ڈاکٹر", "or": "ଡାକ୍ତର", "as": "চিকিৎসক", "sa": "चिकित्सकः", "es": "Médico", "fr": "Médecin"},
    "GOOD MORNING": {"hi": "शुभ प्रभात", "gu": "શુભ સવાર", "mr": "शुभ सकाळ", "bn": "সুপ্রভাত", "ta": "காலை வணக்கம்", "te": "శుభోదయం", "kn": "ಶುಭೋದಯ", "ml": "സുപ്രഭാതം", "pa": "ਸ਼ੁਭ ਸਵੇਰ", "ur": "صبح بخیر", "or": "ଶୁଭ ସକାଳ", "as": "সুপ্ৰভাত", "sa": "सुप्रभातम्", "es": "Buenos días", "fr": "Bonjour"},
    "GOODBYE": {"hi": "अलविदा", "gu": "આવજો", "mr": "निरोप", "bn": "বিদায়", "ta": "விடைபெறுகிறேன்", "te": "వీడ్కోలు", "kn": "ವಿದಾಯ", "ml": "വിട", "pa": "ਅਲਵਿਦਾ", "ur": "الوداع", "or": "ବିଦାୟ", "as": "বিদায়", "sa": "पुनर्मिलामः", "es": "Adiós", "fr": "Au revoir"},
    "LOVE": {"hi": "प्यार", "gu": "પ્રેમ", "mr": "प्रेम", "bn": "ভালোবাসা", "ta": "அன்பு", "te": "ప్రేమ", "kn": "ಪ್ರೀತಿ", "ml": "സ്നേഹം", "pa": "ਪਿਆਰ", "ur": "محبت", "or": "ଭଲପାଇବା", "as": "মৰম", "sa": "प्रेम", "es": "Amor", "fr": "Amour"},
    "FAMILY": {"hi": "परिवार", "gu": "પરિવાર", "mr": "कुटुंब", "bn": "পরিবার", "ta": "குடும்பம்", "te": "కుటుంబం", "kn": "ಕುಟುಂಬ", "ml": "കുടുംബം", "pa": "ਪਰਿਵਾਰ", "ur": "خاندان", "or": "ପରିବାର", "as": "পৰিয়াল", "sa": "परिवारः", "es": "Familia", "fr": "Famille"},
    "FRIEND": {"hi": "दोस्त", "gu": "મિત્ર", "mr": "मित्र", "bn": "বন্ধু", "ta": "நண்பன்", "te": "స్నేహితుడు", "kn": "ಸ್ನೇಹಿತ", "ml": "സുഹൃത്ത്", "pa": "ਦੋਸਤ", "ur": "دوست", "or": "ବନ୍ଧୁ", "as": "বন্ধু", "sa": "मित्रम्", "es": "Amigo", "fr": "Ami"},
    "FATHER": {"hi": "पिता", "gu": "પિતા", "mr": "वडील", "bn": "বাবা", "ta": "அப்பா", "te": "తండ్రి", "kn": "ತಂದೆ", "ml": "അച്ഛൻ", "pa": "ਪਿਤਾ", "ur": "والد", "or": "ବାପା", "as": "দেউতা", "sa": "पिता", "es": "Padre", "fr": "Père"},
    "MOTHER": {"hi": "माँ", "gu": "માતા", "mr": "आई", "bn": "মা", "ta": "அம்மா", "te": "తల్లి", "kn": "ತಾಯಿ", "ml": "അമ്മ", "pa": "ਮਾਂ", "ur": "ماں", "or": "ମା", "as": "মা", "sa": "माता", "es": "Madre", "fr": "Mère"},
    "EMERGENCY": {"hi": "आपातकाल", "gu": "કટોકટી", "mr": "आणीबाणी", "bn": "জরুরী", "ta": "அவசரம்", "te": "அత్యవసరం", "kn": "ತುರ್ತು", "ml": "അടിയന്തരാവസ്ഥ", "pa": "ਸੰਕਟਕਾਲ", "ur": "ہنگامی صورتحال", "or": "ଜରୁରୀକାଳୀନ", "as": "ଜৰୁৰীকালীন", "sa": "आपत्कालः", "es": "Emergencia", "fr": "Urgence"},
    "HOSPITAL": {"hi": "अस्पताल", "gu": "હોસ્પિટલ", "mr": "रुग्णालय", "bn": "হাসপাতাল", "ta": "மருத்துவமனை", "te": "ఆసుపత్రి", "kn": "ಆಸ್ಪತ್ರೆ", "ml": "ആശുപത്രി", "pa": "ਹਸਪਤਾਲ", "ur": "ہسپتال", "or": "ଡାକ୍ତରଖାନା", "as": "চিকিৎসালয়", "sa": "चिकित्सालयः", "es": "Hospital", "fr": "Hôpital"},
    "STOP": {"hi": "रुको", "gu": "રોકો", "mr": "थांबा", "bn": "থামুন", "ta": "நிறுத்து", "te": "ఆగు", "kn": "ನಿಲ್ಲಿಸಿ", "ml": "നിർത്തുക", "pa": "ਰੁਕੋ", "ur": "روکیں", "or": "ଅଟକନ୍ତୁ", "as": "ৰওক", "sa": "तिष्ठ", "es": "Alto", "fr": "Arrêt"},
    "WELCOME": {"hi": "स्वागत है", "gu": "સ્વાગત છે", "mr": "स्वागत आहे", "bn": "স্বাগতম", "ta": "வரவேற்கிறோம்", "te": "స్వాగతం", "kn": "ಸ್ವಾಗತ", "ml": "സ്വാഗതം", "pa": "ਸੁਆਗਤ ਹੈ", "ur": "خوش آمدید", "or": "ସ୍ୱାଗତ", "as": "স্বাগতম", "sa": "स्वागतम्", "es": "Bienvenido", "fr": "Bienvenue"}
}


# =====================================================
# Supported Languages List
# =====================================================

@app.get("/languages")
async def get_supported_languages():
    """Return all supported Indian and global languages."""
    return {
        "total_languages": len(LANGUAGE_CONFIG),
        "languages": LANGUAGE_CONFIG
    }


# =====================================================
# Translation Endpoint
# =====================================================

@app.post("/translate")
async def translate_text(data: dict):
    """
    Translate text from English into selected Indian or global language.
    Expected: {"text": "HELLO", "target_lang": "hi"}
    """
    text = data.get("text", "").strip()
    target_lang = data.get("target_lang", "en").lower().strip()

    if not text:
        return {"original": "", "translated": "", "target_lang": target_lang}

    if target_lang == "en":
        return {"original": text, "translated": text, "target_lang": "en"}

    upper_text = text.upper()

    # 1. Check direct dictionary
    if upper_text in TRANSLATION_DICTIONARY and target_lang in TRANSLATION_DICTIONARY[upper_text]:
        return {
            "original": text,
            "translated": TRANSLATION_DICTIONARY[upper_text][target_lang],
            "target_lang": target_lang,
            "source": "dictionary"
        }

    # 2. Try online Google Translate
    try:
        encoded = urllib.parse.quote(text)
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl={target_lang}&dt=t&q={encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data and data[0] and data[0][0]:
                translated = "".join(item[0] for item in data[0] if item and item[0])
                return {
                    "original": text,
                    "translated": translated,
                    "target_lang": target_lang,
                    "source": "online"
                }
    except Exception as e:
        logger.warning(f"Google translate error: {e}")

    # 3. Fallback to word-by-word dictionary
    words = upper_text.split()
    translated_words = []
    for w in words:
        if w in TRANSLATION_DICTIONARY and target_lang in TRANSLATION_DICTIONARY[w]:
            translated_words.append(TRANSLATION_DICTIONARY[w][target_lang])
        else:
            translated_words.append(w)

    return {
        "original": text,
        "translated": " ".join(translated_words) if translated_words else text,
        "target_lang": target_lang,
        "source": "fallback"
    }


# =====================================================
# Batch Prediction
# =====================================================

@app.post("/batch-predict")
async def batch_predict(
    data: dict
):

    """
    Process multiple base64 encoded images.

    Expected:
        {
            "frames": [
                "base64_image1",
                "base64_image2"
            ]
        }
    """

    if not predictor or not predictor.model:

        return JSONResponse(
            status_code=500,
            content={
                "error": "Predictor not initialized"
            }
        )

    try:

        frames_data = data.get(
            "frames",
            []
        )

        results = []

        for frame_b64 in frames_data:

            try:

                # -----------------------------------------
                # Remove data URI prefix
                # -----------------------------------------

                if "," in frame_b64:

                    frame_b64 = frame_b64.split(
                        ",",
                        1
                    )[1]

                # -----------------------------------------
                # Decode
                # -----------------------------------------

                image_bytes = base64.b64decode(
                    frame_b64
                )

                image = Image.open(
                    BytesIO(image_bytes)
                )

                image = ImageOps.exif_transpose(
                    image
                )

                image = image.convert("RGB")

                # -----------------------------------------
                # Convert to OpenCV
                # -----------------------------------------

                frame = cv2.cvtColor(
                    np.array(image),
                    cv2.COLOR_RGB2BGR
                )

                # -----------------------------------------
                # Predict
                # -----------------------------------------

                result = predictor.predict(
                    frame
                )

                results.append(result)

            except Exception as e:

                logger.error(
                    f"❌ Error processing frame: {e}"
                )

                results.append({
                    "error": str(e)
                })

        return {
            "total_frames": len(frames_data),
            "results": results
        }

    except Exception as e:

        logger.exception(
            "❌ Error in /batch-predict"
        )

        return JSONResponse(
            status_code=400,
            content={
                "error": str(e)
            }
        )


# =====================================================
# Model Information
# =====================================================

@app.get("/model-info")
async def model_info():

    if not predictor or not predictor.model:

        return {
            "status": "not_loaded",
            "message": "Model not initialized"
        }

    try:

        return {
            "status": "loaded",
            "model_type": type(
                predictor.model
            ).__name__,
            "classes": list(
                predictor.model.classes_
            ),
            "n_classes": len(
                predictor.model.classes_
            ),
            "model_source": predictor.model_source,
            "model_path": MODEL_PATH,
            "n_features": (
                predictor.model.n_features_in_
                if hasattr(
                    predictor.model,
                    "n_features_in_"
                )
                else "unknown"
            )
        }

    except Exception as e:

        logger.error(
            f"❌ Error getting model info: {e}"
        )

        return {
            "status": "error",
            "message": str(e)
        }


# =====================================================
# Error Handler
# =====================================================

@app.exception_handler(Exception)
async def global_exception_handler(
    request,
    exc
):

    logger.error(
        f"❌ Unhandled exception: {exc}"
    )

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

    logger.info(
        "🚀 SignCare AI API Starting..."
    )

    if predictor and predictor.model:

        logger.info(
            f"✅ Model loaded with "
            f"{len(predictor.model.classes_)} classes"
        )

        logger.info(
            f"✅ Dictionary loaded with "
            f"{len(WORDS)} words"
        )

    else:

        logger.warning(
            "⚠️ Warning: Model not properly initialized"
        )


# =====================================================
# Shutdown Event
# =====================================================

@app.on_event("shutdown")
async def shutdown_event():

    logger.info(
        "🛑 SignCare AI API Shutting down..."
    )


# =====================================================
# Run FastAPI Server
# =====================================================

if __name__ == "__main__":

    import uvicorn

    logger.info(
        "🚀 Starting SignCare AI FastAPI Server"
    )

    logger.info(
        "📍 Server running at: "
        "http://localhost:8000"
    )

    logger.info(
        "📚 API Documentation: "
        "http://localhost:8000/docs"
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )