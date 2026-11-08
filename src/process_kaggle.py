"""Convert the Kaggle ASL image folders into wrist-relative landmark rows."""

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support
import os
from pathlib import Path
import tempfile

import cv2
import mediapipe as mp
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = (
    BASE_DIR
    / "archive (11)"
    / "asl_alphabet_train"
    / "asl_alphabet_train"
)
OUTPUT_PATH = BASE_DIR / "kaggle_augmented_data.csv"
PROGRESS_INTERVAL = 500
MAX_WORKERS = 2
LANDMARK_COUNT = 21
FEATURE_COUNT = LANDMARK_COUNT * 3


def extract_wrist_relative_landmarks(results):
    if not results.multi_hand_landmarks:
        return None

    landmarks = results.multi_hand_landmarks[0].landmark
    wrist = np.array(
        [landmarks[0].x, landmarks[0].y, landmarks[0].z],
        dtype=np.float32,
    )
    coordinates = np.asarray(
        [[landmark.x, landmark.y, landmark.z] for landmark in landmarks],
        dtype=np.float32,
    )
    normalized = coordinates - wrist
    if normalized.shape != (LANDMARK_COUNT, 3) or not np.isfinite(normalized).all():
        return None
    return normalized.reshape(FEATURE_COUNT)


def process_class_folder(class_directory, temporary_csv_path):
    image_paths = sorted(
        (
            path
            for path in class_directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".jpg"
        ),
        key=lambda path: path.name.lower(),
    )
    processed_images = 0
    original_rows = 0
    mirrored_rows = 0
    unreadable_images = 0
    no_hand_detections = 0

    print(
        f"Processing folder {class_directory.name}: "
        f"{len(image_paths):,} images",
        flush=True,
    )
    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    ) as hands, temporary_csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.writer(csv_file)

        for image_path in image_paths:
            bgr_image = cv2.imread(str(image_path))
            if bgr_image is None:
                unreadable_images += 1
                processed_images += 1
                continue

            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            original_results = hands.process(rgb_image)
            original_features = extract_wrist_relative_landmarks(original_results)
            if original_features is not None:
                writer.writerow(
                    [*original_features.tolist(), class_directory.name]
                )
                original_rows += 1
            else:
                no_hand_detections += 1

            mirrored_rgb_image = cv2.flip(rgb_image, 1)
            mirrored_results = hands.process(mirrored_rgb_image)
            mirrored_features = extract_wrist_relative_landmarks(mirrored_results)
            if mirrored_features is not None:
                writer.writerow(
                    [*mirrored_features.tolist(), class_directory.name]
                )
                mirrored_rows += 1
            else:
                no_hand_detections += 1

            processed_images += 1
            if processed_images % PROGRESS_INTERVAL == 0:
                csv_file.flush()
                print(
                    f"  {class_directory.name}: "
                    f"{processed_images:,}/{len(image_paths):,} images "
                    f"({original_rows + mirrored_rows:,} rows)",
                    flush=True,
                )

    rows_written = original_rows + mirrored_rows
    print(
        f"Finished folder {class_directory.name}: "
        f"{processed_images:,} images, {rows_written:,} rows",
        flush=True,
    )
    return {
        "class_name": class_directory.name,
        "csv_path": temporary_csv_path,
        "processed_images": processed_images,
        "original_rows": original_rows,
        "mirrored_rows": mirrored_rows,
        "unreadable_images": unreadable_images,
        "no_hand_detections": no_hand_detections,
    }


def main():
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {INPUT_DIR}")

    class_directories = sorted(
        (path for path in INPUT_DIR.iterdir() if path.is_dir()),
        key=lambda path: path.name.lower(),
    )
    if not class_directories:
        raise ValueError(f"No label folders found in {INPUT_DIR}")

    total_images = sum(
        1
        for directory in class_directories
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".jpg"
    )
    if total_images == 0:
        raise ValueError(f"No .jpg images found in {INPUT_DIR}")

    headers = [f"feature_{index}" for index in range(FEATURE_COUNT)] + ["label"]
    workers = min(MAX_WORKERS, os.cpu_count() or 1, len(class_directories))
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_PATH}")
    print(
        f"Classes: {len(class_directories)}; images: {total_images:,}; "
        f"workers: {workers}"
    )

    with tempfile.TemporaryDirectory(
        prefix=".process_kaggle_",
        dir=BASE_DIR,
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        futures = []
        results = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for class_directory in class_directories:
                class_csv = temporary_root / f"{class_directory.name}.csv"
                futures.append(
                    executor.submit(
                        process_class_folder,
                        class_directory,
                        class_csv,
                    )
                )
            for future in as_completed(futures):
                results.append(future.result())

        results.sort(key=lambda result: result["class_name"].lower())
        temporary_output = temporary_root / OUTPUT_PATH.name
        with temporary_output.open(
            "w", newline="", encoding="utf-8"
        ) as output_file:
            writer = csv.writer(output_file)
            writer.writerow(headers)
            for result in results:
                with Path(result["csv_path"]).open(
                    "r", newline="", encoding="utf-8"
                ) as class_file:
                    output_file.write(class_file.read())
            output_file.flush()
            os.fsync(output_file.fileno())

        temporary_output.replace(OUTPUT_PATH)

    processed_images = sum(result["processed_images"] for result in results)
    original_rows = sum(result["original_rows"] for result in results)
    mirrored_rows = sum(result["mirrored_rows"] for result in results)
    unreadable_images = sum(result["unreadable_images"] for result in results)
    no_hand_detections = sum(result["no_hand_detections"] for result in results)
    print("Processing complete.")
    print(f"  Images processed: {processed_images:,}/{total_images:,}")
    print(f"  Original-orientation rows: {original_rows:,}")
    print(f"  Mirrored-orientation rows: {mirrored_rows:,}")
    print(f"  Total CSV data rows: {original_rows + mirrored_rows:,}")
    print(f"  Images without a detected hand: {no_hand_detections:,}")
    print(f"  Unreadable images: {unreadable_images:,}")
    print(f"  CSV saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    freeze_support()
    main()
