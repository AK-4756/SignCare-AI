import csv
import os
import time

import cv2

try:
    import mediapipe as mp
except Exception as exc:
    mp = None
    mediapipe_import_error = exc
else:
    mediapipe_import_error = None


class _FallbackHandProcessor:
    def process(self, _frame):
        return type("DummyResults", (), {"multi_hand_landmarks": None})()


class _FallbackDrawingUtils:
    @staticmethod
    def draw_landmarks(*args, **kwargs):
        return None


if mp is not None:
    try:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        mp_draw = mp.solutions.drawing_utils
        hand_tracking_enabled = True
    except Exception as exc:
        mediapipe_import_error = exc
        mp_hands = type("FallbackHandsModule", (), {"HAND_CONNECTIONS": []})()
        hands = _FallbackHandProcessor()
        mp_draw = _FallbackDrawingUtils()
        hand_tracking_enabled = False
else:
    mp_hands = type("FallbackHandsModule", (), {"HAND_CONNECTIONS": []})()
    hands = _FallbackHandProcessor()
    mp_draw = _FallbackDrawingUtils()
    hand_tracking_enabled = False

if not hand_tracking_enabled:
    print(
        "MediaPipe hand tracking unavailable: "
        f"{mediapipe_import_error}. Continuing without landmarks."
    )


DATA_DIR = "Data"
CSV_PATH = os.path.join(DATA_DIR, "alphabet_data.csv")
MAX_SAMPLES = 600


def ensure_csv_exists():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        header = [
            f"p_{index}_{coordinate}"
            for index in range(21)
            for coordinate in ("x", "y", "z")
        ] + ["label"]
        with open(CSV_PATH, mode="w", newline="", encoding="utf-8") as csv_file:
            csv.writer(csv_file).writerow(header)


def initialize_camera():
    for index in (0, 1):
        print(f"Attempting to open camera index {index}...")
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        time.sleep(2)

        if capture.isOpened():
            ret = False
            for _ in range(5):
                ret, _frame = capture.read()
            if ret:
                print(f"Camera index {index} is working.")
                return capture
            capture.release()

    return None


def main():
    if not hand_tracking_enabled:
        raise RuntimeError(
            "MediaPipe hand tracking is unavailable, so no training landmarks "
            "can be collected."
        ) from mediapipe_import_error

    ensure_csv_exists()
    capture = initialize_camera()
    if capture is None:
        raise RuntimeError("Could not open any camera.")

    try:
        target_sign = input(
            "Which letter/sign are you recording (for example A, M, N, R, T, U, V)? "
        ).strip().upper()
        if not target_sign:
            raise ValueError("The sign label cannot be empty.")

        print(f"Get your '{target_sign}' hand sign ready.")
        start_time = time.time()
        while time.time() - start_time < 5:
            ret, frame = capture.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                    )

            countdown = int(5 - (time.time() - start_time)) + 1
            cv2.putText(
                frame,
                f"GET READY: {countdown}",
                (width // 2 - 150, height // 2),
                cv2.FONT_HERSHEY_DUPLEX,
                1.5,
                (0, 255, 255),
                3,
            )
            cv2.imshow("SignCare Data Collection", frame)
            cv2.setWindowProperty(
                "SignCare Data Collection",
                cv2.WND_PROP_TOPMOST,
                1,
            )
            cv2.waitKey(1)

        print(f"Recording '{target_sign}'...")
        count = 0
        with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            while count < MAX_SAMPLES:
                ret, frame = capture.read()
                if not ret:
                    print("Camera frame could not be read; stopping capture.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                        )

                        wrist = hand_landmarks.landmark[0]
                        row = []
                        for landmark in hand_landmarks.landmark:
                            row.extend(
                                [
                                    landmark.x - wrist.x,
                                    landmark.y - wrist.y,
                                    landmark.z - wrist.z,
                                ]
                            )
                        row.append(target_sign)
                        writer.writerow(row)
                        count += 1
                        if count % 10 == 0:
                            csv_file.flush()
                            print(f"Captured: {count}/{MAX_SAMPLES}")
                        if count >= MAX_SAMPLES:
                            break

                cv2.putText(
                    frame,
                    f"COLLECTING {target_sign}: {count}/{MAX_SAMPLES}",
                    (15, 30),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("SignCare Data Collection", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Capture cancelled by user.")
                    break

        print(f"\nFinished recording {target_sign}: {count} samples saved.")
        print(f"CSV: {os.path.abspath(CSV_PATH)}")
    finally:
        capture.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()
