# SignCare AI: Code-Verified Paper Facts

Scope: current workspace source and artifacts, plus read-only runtime probes of the project `.venv` and Windows system metadata. No application or data files were changed to produce these facts. Line citations below refer to the current source files.

## 1. Library and platform versions

### Project virtual environment

`.\.venv\Scripts\python.exe --version` reported **Python 3.10.11**. This is the environment whose packages were measured with `.\.venv\Scripts\python.exe -m pip list --format=freeze`.

| Component | `requirements.txt` pin | Installed `.venv` distribution / runtime |
|---|---|---|
| Python | Not pinned | Python 3.10.11 |
| FastAPI | 0.104.1 | 0.104.1 |
| Uvicorn | 0.24.0 | 0.24.0 |
| OpenCV | `opencv-python==4.8.1.78` | Distribution `opencv-python==4.8.1.78`; `cv2.__version__` reports **4.11.0** |
| Pillow | 10.0.1 | 10.0.1 |
| MediaPipe | 0.10.5 | 0.10.5 |
| scikit-learn | 1.3.2 | 1.3.2 |
| NumPy | 1.24.3 | 1.24.3 |
| pandas | 2.0.3 | 2.0.3 |
| python-multipart | 0.0.6 | 0.0.6 |
| customtkinter | 5.2.0 | 5.2.0 |
| pyttsx3 | 2.90 | 2.90 |
| deep-translator | 1.11.4 | 1.11.4 |
| SpeechRecognition | 3.10.0 | 3.10.0 |
| python-dotenv | 1.0.0 | 1.0.0 |

These are the direct requirements and pins in [requirements.txt](./requirements.txt) (lines 2–29).

Other installed distributions used by the backend/framework or imported training module include `starlette==0.27.0`, `pydantic==2.14.1`, `pydantic_core==2.50.1`, `scipy==1.15.3`, `protobuf==3.20.3`, `joblib==1.6.0`, and `threadpoolctl==3.7.0`. `app.py` directly imports FastAPI, OpenCV, NumPy, MediaPipe, Pillow, and `motion_recognition`; `motion_recognition.py` imports NumPy and scikit-learn. Uvicorn is imported when `app.py` is run as a script. See [app.py](./app.py) lines 6–35 and line 1685, and [motion_recognition.py](./motion_recognition.py) lines 3–10.

**Version discrepancy:** `opencv-python==4.8.1.78` is installed as declared, but the imported `cv2` module reports 4.11.0. The environment also contains `opencv-contrib-python==4.11.0.86`, which is not listed in `requirements.txt`. Therefore the runtime OpenCV module version differs from the declared `opencv-python` version. The other pinned distributions listed above matched their requirements.

VS Code/Pylance reports its *selected* interpreter as `C:\Users\hp\AppData\Local\Programs\Python\Python313\python.exe`, not the project `.venv`. The project `.venv` runtime used for this report is still Python 3.10.11. The sibling `.\venv\Scripts\python.exe` cannot run: its launcher reports that its configured Python 3.11 path under another Windows user does not exist. See the `pip list` and interpreter probe results; these are runtime facts and do not have source-code line citations.

### Windows identification

- `Win32_OperatingSystem` reported `Microsoft Windows 11 Pro`, version `10.0.26300`, build `26300`, 64-bit.
- The Windows `CurrentVersion` registry key reported `ProductName=Windows 10 Pro`, `DisplayVersion=26H2`, `CurrentBuild=26300`, `UBR=9711`, and `EditionID=Professional`.

The two system interfaces return conflicting product names (Windows 11 Pro vs Windows 10 Pro). Exact marketing edition/version **cannot be determined consistently** from these conflicting identifiers. The raw reported identifiers are recorded here rather than reconciling them by assumption.

## 2. `/predict-base64` preprocessing, hand validation, and temporal checks

### Processing order

1. `/predict-base64` reads `data["image"]` and strips a data-URI prefix if present ([app.py](./app.py), line 1168 and lines 1180–1187).
2. It Base64-decodes the image, opens it with Pillow, applies EXIF orientation, and converts it to RGB ([app.py](./app.py), lines 1191–1208).
3. It converts the RGB NumPy image to OpenCV BGR using `cv2.COLOR_RGB2BGR` ([app.py](./app.py), lines 1210–1215).
4. It calls `predictor.predict(frame)` ([app.py](./app.py), lines 1217–1221). `preprocess_frame` resizes the frame to **640 pixels wide**, sets height to `int(original_height * 640 / original_width)`, and thus preserves aspect ratio up to integer rounding ([app.py](./app.py), lines 240–258).
5. `extract_landmarks` converts that resized BGR frame to RGB using `cv2.COLOR_BGR2RGB` before MediaPipe processing ([app.py](./app.py), lines 345–363).

The browser separately draws each inference frame to a 640-pixel-wide canvas before encoding it as JPEG; it computes height with the original video aspect ratio ([signcare-ai.html](./signcare-ai.html), lines 1620–1647). The backend still applies its own 640-pixel-width resize.

### MediaPipe Hands configuration

`app.py` explicitly sets `static_image_mode=False`, `max_num_hands=1`, `min_detection_confidence=0.6`, and `min_tracking_confidence=0.45` ([app.py](./app.py), lines 155–180). It does not pass `model_complexity`. In the installed MediaPipe 0.10.5 runtime, the `Hands` constructor signature reports the default `model_complexity=1`; therefore the effective complexity is **1 by library default**, not an explicit setting in `app.py`.

### Landmark validation checks in execution order

`validate_hand_landmarks` ([app.py](./app.py), lines 265–339) rejects a candidate if any of the following hold:

1. Landmark collection is empty or does not contain exactly **21** landmarks (lines 273–274).
2. Any landmark lacks an `x`, `y`, or `z` coordinate (lines 280–286).
3. Any coordinate is non-finite (lines 288–289).
4. Any `x` or `y` is outside **[-0.20, 1.20]** (lines 291–293). There is no separate numeric range check for `z`; `z` is checked for presence and finiteness.
5. The bounding-box width is below **0.05**, height below **0.05**, or diagonal span below **0.08** (configuration at lines 121–128; rejection at lines 299–307).
6. **Both** width and height exceed **0.98**. The code uses `w > MAX_HAND_SIZE and h > MAX_HAND_SIZE`, not an OR condition (lines 309–310).
7. Width/height aspect ratio is below **0.15** or above **5.0**; denominator is clamped with `max(h, 1e-6)` (configuration at lines 130–133; check at lines 312–315).
8. Both x-coordinate and y-coordinate standard deviations are below **0.012** (configuration at lines 135–137; check at lines 317–321).
9. The larger of wrist-to-middle-MCP (landmark 9) and wrist-to-index-MCP (landmark 5) 2D distances is below **0.025** (configuration at lines 139–141; calculation/check at lines 323–332).
10. The maximum wrist-to-fingertip 2D distance for landmarks **4, 8, 12, 16, 20** is below **0.035** (configuration at lines 142–143; check at lines 334–337).

### Temporal confirmation and movement

- `REQUIRED_VALID_FRAMES=2`; the predictor marks a hand confirmed when `consecutive_valid_frames >= 2` ([app.py](./app.py), lines 149–153 and 491–495).
- A center-to-center movement greater than **0.45** normalized image units resets the count to one and returns a no-detection result for that frame when more than one valid frame is required ([app.py](./app.py), lines 460–482).
- A missing/invalid frame increments the missing counter. The confirmation state is cleared when the counter reaches **2**; one missing frame alone does not immediately clear it ([app.py](./app.py), lines 443–455). Consequently, the code’s threshold is two valid detections, but a single missing frame can occur between them without resetting the existing count.

### Model path

The static predictor’s configured path is `model/sign_language_model_combined.p` ([app.py](./app.py), line 86). Its loader reads that file with `pickle.load` ([app.py](./app.py), lines 206–224).

## 3. Frontend recognition rules

- For the ordinary static-letter branch, the same predicted letter must be observed at least **3** times (`stableCount >= 3`), and the backend probability confidence must be at least **0.60** (60%) before the frontend appends it ([signcare-ai.html](./signcare-ai.html), lines 1776–1807).
- If the motion model is ready, the ordinary static-letter branch also requires an `OTHER` motion prediction with confidence at least **0.55**; otherwise the static letter is not appended ([signcare-ai.html](./signcare-ai.html), lines 1801–1804).
- The separate moving-J/Z branch appends J or Z at motion confidence **>= 0.70** without the three-static-prediction rule ([signcare-ai.html](./signcare-ai.html), lines 1752–1758 and 1790–1796).
- The static prediction timer is **100 ms** ([signcare-ai.html](./signcare-ai.html), lines 1948–1952).
- `getUserMedia` requests user-facing video with ideal dimensions **1280×720** and no audio ([signcare-ai.html](./signcare-ai.html), lines 1874–1891). These are requested ideal dimensions; the actual camera stream resolution **cannot be determined from the request alone**.

## 4. Current and split-first model artifacts

### Paths and formats

- Before approval, `app.py` pointed to `model/sign_language_model_combined.p`. After approval, the current `app.py` path is `model/sign_language_model_combined_splitfirst.p` ([app.py](./app.py), line 86).
- Split-first artifact: `model/sign_language_model_combined_splitfirst.p`; its output path is declared in [evaluate_no_leakage.py](./evaluate_no_leakage.py), line 23.
- Both artifacts were read from disk with pickle. Both artifacts are dictionaries containing a `"model"` key and a 29-class, 63-feature Random Forest. The former model metadata records `test_accuracy=0.9933641013`; the split-first artifact records `test_accuracy=0.9899873524` and `test_macro_f1=0.9889914359`.
- The split-first artifact’s fit metadata records **489.607 seconds**. Its dictionary structure and metadata fields are defined in [evaluate_no_leakage.py](./evaluate_no_leakage.py), lines 296–312.

The `app.py` loader already accepts this artifact format: when the loaded value is a dictionary containing `"model"`, it assigns `model_dict["model"]` to `self.model` ([app.py](./app.py), lines 216–224). **No loader-format code change is needed** for the split-first artifact.

### Approved path change

With user approval, only `MODEL_PATH` at [app.py](./app.py), line 86, was changed from `sign_language_model_combined.p` to `sign_language_model_combined_splitfirst.p`. The app must be restarted to load the selected artifact.

On that path change:

- The application now selects the split-first Random Forest for predictions instead of the former artifact; its class count and feature count remain 29 and 63.
- The API’s model source/path metadata identifies the selected artifact after restart ([app.py](./app.py), lines 1567–1597).
- Paper results attributed to the selected model should use that artifact’s measured split-first evaluation numbers (**98.9987% accuracy**, **0.988991 macro-F1**), not the former artifact’s stored test-accuracy value (**99.3364%**). The former artifact’s value comes from the post-oversampling random split pipeline in [train_model.py](./train_model.py), lines 46–61; the split-first artifact is evaluated before training-row-only repetition in [evaluate_no_leakage.py](./evaluate_no_leakage.py), lines 275–304.
- The actual live accuracy after deployment **cannot be determined from the file-path change**; it requires a new held-out or live evaluation.

## 5. Checks of the five paper statements

| Statement | Finding | Source evidence |
|---|---|---|
| (a) Feature vector is wrist-relative with 63 values. | **Supported.** Extraction subtracts wrist x/y/z (landmark 0) from each landmark and checks for 63 values. | [app.py](./app.py), lines 345–355, 382–410, 511–513. |
| (b) J and Z are predicted as static classes by the main model. | **Supported, with a separate motion route also present.** The static model is queried for its predicted class and its artifacts contain J and Z among 29 classes. The frontend additionally uses a separate sequence model for moving J/Z predictions. | [app.py](./app.py), lines 519–540; runtime inspection of both model artifacts; [motion_recognition.py](./motion_recognition.py), lines 13–16; [signcare-ai.html](./signcare-ai.html), lines 1752–1769. |
| (c) Translation uses a 21-phrase local dictionary and then online services. | **Supported for non-English text not covered by the local dictionary.** The frontend checks its local mapping, then Google Translate, then MyMemory. The backend checks its local mapping, then Google Translate, then a word-by-word local fallback. Both source dictionaries contain 21 keys, verified by parsing the current source. | Frontend: [signcare-ai.html](./signcare-ai.html), lines 2167–2188 and 2191–2249. Backend: [app.py](./app.py), line 1340 and lines 1383–1437. |
| (d) Suggestion list has 178 entries. | **Supported.** `MULTILINGUAL_WORDS` has 178 entries in the current source; `WORDS` is built from its keys. Count verified by parsing the source dictionary. | [app.py](./app.py), lines 595–596 and 826. |
| (e) History is stored in browser `localStorage`. | **Supported.** Initial state reads `signcare_history`; save writes it, and clear removes it. | [signcare-ai.html](./signcare-ai.html), lines 1115, 2371, and 2412. |

## Runtime measurement notes

These facts came from runtime probes rather than source lines: package versions, the MediaPipe constructor default, Windows identifiers, and pickle artifact metadata. The cited source lines establish the code paths and declared settings; runtime-probe results above identify the environment/artifact values. `clean_data.py` is not part of the inference path; it loads `Data/alphabet_data.csv`, filters out label `P`, and writes the filtered rows back to that CSV ([clean_data.py](./clean_data.py), lines 4–24).
