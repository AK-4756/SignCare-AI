"""Train the webcam motion classifier for ASL J, Z, and OTHER gestures."""

import os

from motion_recognition import train_motion_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data", "motion_sequences")
MODEL_PATH = os.path.join(BASE_DIR, "model", "motion_sequence_model.p")


if __name__ == "__main__":
    try:
        result = train_motion_model(DATA_DIR, MODEL_PATH)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Motion model training failed: {error}") from error

    print(f"Held-out accuracy: {result['accuracy']:.2%}")
    print("Held-out classification report:")
    print(result["report"])
    print(f"Training samples by class: {result['sample_counts']}")
    print(f"Motion model saved to: {result['model_path']}")
