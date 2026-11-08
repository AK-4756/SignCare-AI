import cv2
import mediapipe as mp
import sys

image_path = sys.argv[1]

image = cv2.imread(image_path)

if image is None:
    print("❌ Could not read image")
    sys.exit()

print("Original image:", image.shape)

mp_hands = mp.solutions.hands


def test_image(name, img):
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.2
    ) as hands:

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            print(f"✅ {name}: HAND DETECTED")
            print(
                f"   Landmarks: "
                f"{len(result.multi_hand_landmarks[0].landmark)}"
            )
            return True
        else:
            print(f"❌ {name}: NO HAND")

    return False


# 1. Original
test_image("Original", image)


# 2. Resize
resized = cv2.resize(image, (640, 360))
test_image("Resized", resized)


# 3. Horizontal flip
flipped = cv2.flip(image, 1)
test_image("Flipped", flipped)


# 4. Increase brightness and contrast
enhanced = cv2.convertScaleAbs(
    image,
    alpha=1.5,
    beta=30
)
test_image("Brightness + Contrast", enhanced)


# 5. CLAHE enhancement
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(
    clipLimit=3.0,
    tileGridSize=(8, 8)
)

l = clahe.apply(l)

enhanced_clahe = cv2.merge((l, a, b))
enhanced_clahe = cv2.cvtColor(
    enhanced_clahe,
    cv2.COLOR_LAB2BGR
)

test_image("CLAHE", enhanced_clahe)


print("\n🏁 Diagnostic complete.")