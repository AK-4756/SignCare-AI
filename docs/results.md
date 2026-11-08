# SignCare AI — Measured Results

Measurement date: 2026-11-02

## 1. Merge and training operation order

`merge_oversampled_data.py` performs these operations in order:

1. Reads `Data/alphabet_data.csv`.
2. Reads `kaggle_augmented_data.csv`.
3. Checks for a `label` column and 63 feature columns; aligns Kaggle feature headers to the personal CSV if needed.
4. Creates five copies of the personal dataframe.
5. Concatenates those five copies with the Kaggle dataframe.
6. Replaces positive/negative infinity, coerces feature columns to numeric, trims labels, and drops rows with missing feature values or labels.
7. Saves `combined_training_data_oversampled.csv`.

`train_model.py` then reads the already oversampled file, validates and converts its 63 features, and performs a stratified 80/20 `train_test_split` on that file. **The split occurs after oversampling.** Duplicate rows can therefore occur in both sets.

Reproducing the current script's split (`random_state=42`) yielded **9,780 distinct feature vectors present in both sets** and **14,573 test rows with an exact 63-feature match in training**. Feature matching ignores labels.

## 2. Leakage-controlled evaluation

Ran [evaluate_no_leakage.py](./evaluate_no_leakage.py). It first splits the original personal and Kaggle rows together using five-fold `StratifiedGroupKFold` (the first fold, seed 42). Group keys keep duplicate and horizontal-mirror feature vectors together after rounding coordinates to five decimal places. It then repeats only the personal training rows five times. No duplicate/mirror group crossed the combined train/test boundary (**0 overlapping groups**). The combined training set had **160,095 rows**; the untouched test set had **28,432 rows**.

This prevents exact and rounded-mirror feature leakage, but it does not provide a person-level split: the personal CSV contains no person or session identifiers.

| Evaluation | Training rows | Test rows | Accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| Personal ×5 + Kaggle → combined holdout | 160,095 | 28,432 | 99.0328% | 0.989401 |
| Kaggle → personal holdout | 96,575 | 2,920 | 46.9863% | 0.405031 |
| Personal → Kaggle holdout | 11,680 | 24,144 | 34.7043% | 0.319367 |

The cross-dataset evaluations use A–Z only. Canonical duplicate/mirror overlap across the two sources measured **0 groups**. The combined `nothing` class has only **1** test example; its displayed metrics are based on that single example.

## 3. Dataset row counts

All CSVs had 64 columns: 63 features and one label. The merged row count equals `5 × personal + Kaggle`.

| Label | Kaggle rows | Personal rows | Merged rows |
|---|---:|---:|---:|
| A | 4,363 | 1,000 | 9,363 |
| B | 4,416 | 400 | 6,416 |
| C | 3,975 | 400 | 5,975 |
| D | 4,893 | 600 | 7,893 |
| E | 4,617 | 600 | 7,617 |
| F | 5,755 | 600 | 8,755 |
| G | 4,957 | 600 | 7,957 |
| H | 4,798 | 600 | 7,798 |
| I | 4,776 | 600 | 7,776 |
| J | 5,142 | 600 | 8,142 |
| K | 5,397 | 400 | 7,397 |
| L | 5,035 | 400 | 7,035 |
| M | 3,221 | 400 | 5,221 |
| N | 2,566 | 400 | 4,566 |
| O | 4,540 | 400 | 6,540 |
| P | 4,065 | 600 | 7,065 |
| Q | 4,161 | 600 | 7,161 |
| R | 5,079 | 600 | 8,079 |
| S | 5,104 | 800 | 9,104 |
| T | 4,690 | 600 | 7,690 |
| U | 5,009 | 600 | 8,009 |
| V | 5,080 | 600 | 8,080 |
| W | 4,894 | 600 | 7,894 |
| X | 4,334 | 600 | 7,334 |
| Y | 5,151 | 600 | 8,151 |
| Z | 4,701 | 400 | 6,701 |
| del | 3,484 | 0 | 3,484 |
| nothing | 6 | 0 | 6 |
| space | 3,214 | 0 | 3,214 |
| **Total** | **127,423** | **14,600** | **200,423** |

The personal CSV has only landmark feature columns and `label`; the number of people and capture sessions **cannot be determined** from the available data.

## 4. Random Forest and training time

`train_model.py` configures `RandomForestClassifier` with:

- `n_estimators=100`
- `random_state=42`
- `n_jobs=-1`
- scikit-learn defaults measured in the installed scikit-learn 1.3.2: `criterion=gini`, `max_depth=None`, `max_features=sqrt`, `min_samples_split=2`, `min_samples_leaf=1`, `class_weight=None`, and `bootstrap=True`
- Input feature count: **63**

The evaluation used the same estimator settings except `n_jobs=2`. Measured `fit()` durations:

| Fit | Seconds |
|---|---:|
| Combined leakage-controlled fit | 259.518 |
| Kaggle fit for Kaggle → personal | 201.374 |
| Personal fit for personal → Kaggle | 10.261 |

The historical training duration of the currently saved production model **cannot be determined**; its artifact does not store a training duration.

## 5. Prediction endpoint performance and hardware

Measured `POST /predict-base64` on the local API at `127.0.0.1:8000`, using `A1.jpg` from the Kaggle A folder, five warm-up requests, and 100 sequential measured requests:

- Mean latency: **49.192 ms**
- Median latency: **46.867 ms**
- 95th-percentile latency: **62.504 ms**
- Sequential throughput: **20.328 requests/second**
- Responses with a detected hand: **100/100**
- Backend CPU consumption during the run: **5.62 CPU-seconds over 5.61 wall-seconds (100.2% of one logical core)**
- Backend process working set: **465.3 MiB before**, **504.8 MiB after**

Measured machine: Intel Core i7-10810U, **6 cores / 12 logical processors**, **15.78 GiB visible RAM**; available RAM after the benchmark was **3.25 GiB**. Windows enumerated an `HP HD Camera` with status `OK`. The benchmark did not activate a camera; it used the static JPEG. Which camera a browser selected for live use **cannot be determined** from this endpoint run.

The throughput figure is for one sequential local client; concurrent or network throughput was not measured.

## 6. Word list and translation

- The backend suggestion list contains **178 entries**, defined in the source-code `MULTILINGUAL_WORDS` mapping.
- `LANGUAGE_CONFIG` contains **16 target-language options**.
- The static translation mapping contains **21 phrases** and **315 phrase/target-language entries**.
- The frontend calls Google Translate and MyMemory translation URLs directly. The backend also calls Google Translate through Python's `urllib`; it has a local phrase dictionary and a word-by-word fallback.
- `deep-translator==1.11.4` appears in `requirements.txt`; no use of that package was found in the inspected application files.

## 7. J/Z motion-model data

The trainer requires **30 samples per label** for J, Z, and OTHER. Measured files under `Data/motion_sequences`:

| Label | Samples | Minimum | Status |
|---|---:|---:|---|
| J | 30 | 30 | Minimum met |
| Z | 0 | 30 | Missing |
| OTHER | 0 | 30 | Missing |

`model/motion_sequence_model.p` was not present. The motion model **cannot be trained from the currently stored samples**; Z and OTHER each lack 30 samples.

## 8. Raw evaluation output

Captured standard output from `evaluate_no_leakage.py` (the subsequent cross-source duplicate check found 0 shared groups, so no cross-dataset test rows needed exclusion):

```text
Dataset files and exact row counts
  personal_rows=14600
  kaggle_rows=127423
  merged_oversampled_rows=200423
  merged_matches_5x_personal_plus_kaggle=True

Current train_model.py row-wise split duplicate check
  train_rows=160338 test_rows=40085
  unique_feature_rows_present_in_both=9780
  test_rows_with_exact_feature_match_in_train=14573

Leakage-controlled combined evaluation
  original_rows=142023 unique_duplicate_mirror_groups=134723
  train_rows_before_oversampling=113591 test_rows=28432
  train_personal=11626 train_kaggle=101965
  test_personal=2974 test_kaggle=25458
  train_test_group_overlap=0

Combined train (personal x5 + Kaggle), test (original held-out rows)
  train_rows=160095 test_rows=28432
  training_seconds=259.518
  accuracy=0.990328
  macro_f1=0.989401
  per_class_precision_recall_f1_support:
              precision    recall  f1-score   support

           A     0.9909    0.9963    0.9936      1088
           B     0.9980    0.9980    0.9980       984
           C     0.9913    0.9902    0.9907       917
           D     0.9952    0.9933    0.9942      1042
           E     0.9940    0.9960    0.9950       994
           F     0.9954    0.9954    0.9954      1300
           G     1.0000    0.9945    0.9973      1099
           H     0.9869    0.9925    0.9897      1066
           I     0.9962    0.9879    0.9921      1075
           J     0.9974    0.9930    0.9952      1149
           K     0.9955    0.9964    0.9960      1111
           L     0.9982    0.9991    0.9986      1110
           M     0.9317    0.9694    0.9502       718
           N     0.9710    0.9208    0.9453       619
           O     0.9904    0.9952    0.9928      1039
           P     0.9966    0.9776    0.9870       894
           Q     0.9888    0.9959    0.9923       974
           R     0.9857    0.9865    0.9861      1189
           S     0.9870    0.9965    0.9917      1139
           T     0.9952    0.9876    0.9914      1046
           U     0.9860    0.9749    0.9804      1157
           V     0.9906    0.9966    0.9936      1169
           W     0.9923    0.9894    0.9909      1040
           X     0.9889    0.9949    0.9919       981
           Y     1.0000    0.9974    0.9987      1147
           Z     0.9950    0.9901    0.9925      1006
         del     0.9891    0.9918    0.9904       731
     nothing     1.0000    1.0000    1.0000         1
       space     0.9670    0.9969    0.9817       647

    accuracy                         0.9903     28432
   macro avg     0.9895    0.9895    0.9894     28432
weighted avg     0.9904    0.9903    0.9903     28432

  confusion_matrix_labels=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
[[1084, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 982, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 908, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 6], [0, 0, 0, 1035, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 990, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1294, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 1093, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 1058, 0, 1, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0, 0, 0, 1062, 0, 0, 0, 1, 1, 3, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 3], [0, 0, 0, 0, 0, 0, 0, 2, 0, 1141, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1107, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1109, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 696, 12, 0, 0, 0, 0, 3, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1], [1, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 42, 570, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1034, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 874, 5, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 6], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 970, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 1173, 0, 0, 9, 0, 0, 2, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1135, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [3, 0, 0, 1, 2, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 1, 1, 0, 1, 1033, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 1128, 3, 3, 3, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1165, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 7, 1029, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 976, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1144, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 4, 1, 0, 0, 0, 0, 0, 996, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 725, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 645]]

Cross-dataset grouped holdouts

Train Kaggle grouped training fold; test personal grouped holdout
  train_rows=96575 test_rows=2920
  training_seconds=201.374
  accuracy=0.469863
  macro_f1=0.405031
  per_class_precision_recall_f1_support:
              precision    recall  f1-score   support

           A     0.6410    0.6944    0.6667       180
           B     0.6797    1.0000    0.8093       104
           C     0.0625    0.0143    0.0233        70
           D     0.3942    0.6406    0.4881       128
           E     0.3007    0.7672    0.4320       116
           F     0.9151    0.8083    0.8584       120
           G     0.0000    0.0000    0.0000       112
           H     0.3831    0.8197    0.5222       122
           I     0.6444    1.0000    0.7838       116
           J     0.4478    0.2679    0.3352       112
           K     0.9385    0.9531    0.9457        64
           L     0.6129    0.2317    0.3363        82
           M     0.0000    0.0000    0.0000        92
           N     0.0000    0.0000    0.0000        70
           O     0.4595    0.5667    0.5075        90
           P     0.0000    0.0000    0.0000       142
           Q     0.8250    0.2946    0.4342       112
           R     0.4606    0.9733    0.6253       150
           S     0.2538    0.3425    0.2915       146
           T     0.5800    0.2589    0.3580       112
           U     0.0476    0.0081    0.0138       124
           V     0.6909    0.3016    0.4199       126
           W     0.9645    0.9855    0.9749       138
           X     0.2168    0.2583    0.2357       120
           Y     1.0000    0.2768    0.4336       112
           Z     0.0377    0.0333    0.0354        60

    accuracy                         0.4699      2920
   macro avg     0.4445    0.4422    0.4050      2920
weighted avg     0.4585    0.4699    0.4248      2920


Train personal grouped training fold; test Kaggle grouped holdout
  train_rows=11680 test_rows=24144
  training_seconds=10.261
  accuracy=0.347043
  macro_f1=0.319367
  per_class_precision_recall_f1_support:
              precision    recall  f1-score   support

           A     0.4214    0.3945    0.4075       877
           B     0.7539    0.3707    0.4971       909
           C     0.1178    0.0812    0.0961       825
           D     0.2780    0.4230    0.3355       981
           E     0.3990    0.5638    0.4673       862
           F     0.7097    0.8158    0.7591      1178
           G     0.0000    0.0000    0.0000       990
           H     0.6613    0.3320    0.4421       994
           I     0.5924    0.6406    0.6155       971
           J     0.0307    0.0088    0.0137      1023
           K     0.4751    0.4167    0.4440      1075
           L     0.4083    0.1150    0.1795      1026
           M     0.0088    0.0048    0.0062       623
           N     0.0012    0.0036    0.0018       549
           O     0.4408    0.2686    0.3338       901
           P     0.1556    0.7146    0.2555       785
           Q     0.3064    0.2032    0.2443       822
           R     0.4545    0.5240    0.4868      1040
           S     0.1053    0.0452    0.0633      1017
           T     0.2380    0.4466    0.3105       909
           U     0.1506    0.1731    0.1610      1017
           V     0.3587    0.2530    0.2967       988
           W     0.9955    0.9272    0.9601       961
           X     0.0000    0.0000    0.0000       844
           Y     0.7786    0.7830    0.7808      1060
           Z     0.1615    0.1320    0.1453       917

    accuracy                         0.3470     24144
   macro avg     0.3463    0.3324    0.3194     24144
weighted avg     0.3649    0.3470    0.3366     24144

Evaluation model parameters
{'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': 2, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}
```

Raw endpoint benchmark output:

```json
{"hand_detected_responses": 100, "image": "archive (11)\\asl_alphabet_train\\asl_alphabet_train\\A\\A1.jpg", "mean_latency_ms": 49.1921860000366, "measured_requests": 100, "median_latency_ms": 46.86690000016824, "p95_latency_ms": 62.503500001184875, "throughput_requests_per_second": 20.32760166112342, "wall_seconds": 4.919419500001823, "warmup_requests": 5}
```

```text
BenchmarkWallSeconds      : 5.61
ServerCPUSecondsDelta     : 5.62
ServerCPUPercentOfOneCore : 100.2
ServerWorkingSetBeforeMiB : 465.3
ServerWorkingSetAfterMiB  : 504.8
FreeRAMGiB                : 3.25
```

## 9. Five-fold combined A-Z and palm-scale ablation (2026-11-02)

Five `StratifiedGroupKFold` folds, seed 42. Duplicate and horizontal-mirror rows were grouped before splitting; personal data was repeated 5× only inside each training fold. Test rows were not oversampled. The scale ablation divides each sample's 63 wrist-relative coordinates by its Euclidean wrist-to-middle-finger-MCP (landmark 9) distance.

| Variant | Accuracy, mean ± sample SD | Macro-F1, mean ± sample SD |
|---|---:|---:|
| Raw wrist-relative features | 0.990201 ± 0.000668 | 0.988919 ± 0.000697 |
| Palm-scale-normalized features | 0.989462 ± 0.000867 | 0.988142 ± 0.000843 |

Fold values:

| Fold | Raw accuracy | Raw macro-F1 | Scale accuracy | Scale macro-F1 |
|---:|---:|---:|---:|---:|
| 1 | 0.989966 | 0.988643 | 0.989265 | 0.987754 |
| 2 | 0.991225 | 0.990008 | 0.990336 | 0.989123 |
| 3 | 0.990481 | 0.989199 | 0.990038 | 0.988628 |
| 4 | 0.989794 | 0.988311 | 0.989573 | 0.988279 |
| 5 | 0.989540 | 0.988436 | 0.988098 | 0.986927 |

## 10. Kaggle MediaPipe detection misses by class

The generated CSV retains successful landmark rows but not source-image IDs. The table counts failed extraction attempts across the original and mirrored pass: `2 × JPG count − successful CSV rows`. Each class contains 3,000 source JPGs, hence 6,000 passes. A unique-image count (images with one or both passes failing) **cannot be determined** from this CSV.

| Class | JPG images | Successful passes | Failed passes |
|---|---:|---:|---:|
| A | 3,000 | 4,363 | 1,637 |
| B | 3,000 | 4,416 | 1,584 |
| C | 3,000 | 3,975 | 2,025 |
| D | 3,000 | 4,893 | 1,107 |
| del | 3,000 | 3,484 | 2,516 |
| E | 3,000 | 4,617 | 1,383 |
| F | 3,000 | 5,755 | 245 |
| G | 3,000 | 4,957 | 1,043 |
| H | 3,000 | 4,798 | 1,202 |
| I | 3,000 | 4,776 | 1,224 |
| J | 3,000 | 5,142 | 858 |
| K | 3,000 | 5,397 | 603 |
| L | 3,000 | 5,035 | 965 |
| M | 3,000 | 3,221 | 2,779 |
| N | 3,000 | 2,566 | 3,434 |
| nothing | 3,000 | 6 | 5,994 |
| O | 3,000 | 4,540 | 1,460 |
| P | 3,000 | 4,065 | 1,935 |
| Q | 3,000 | 4,161 | 1,839 |
| R | 3,000 | 5,079 | 921 |
| S | 3,000 | 5,104 | 896 |
| space | 3,000 | 3,214 | 2,786 |
| T | 3,000 | 4,690 | 1,310 |
| U | 3,000 | 5,009 | 991 |
| V | 3,000 | 5,080 | 920 |
| W | 3,000 | 4,894 | 1,106 |
| X | 3,000 | 4,334 | 1,666 |
| Y | 3,000 | 5,151 | 849 |
| Z | 3,000 | 4,701 | 1,299 |
| **Total** | **87,000** | **127,423** | **46,577** |

## 11. Palm-scale cross-dataset ablation

These are the first grouped holdout folds (seed 42), not five-fold averages. Cross-source duplicate/mirror groups measured: **0**.

| Train → test | Feature variant | Train rows | Test rows | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|---:|
| Kaggle → personal | Raw | 96,575 | 2,920 | 0.435274 | 0.392284 |
| Kaggle → personal | Palm-scale normalized | 96,575 | 2,920 | 0.511986 | 0.476805 |
| Personal → Kaggle | Raw | 11,680 | 24,144 | 0.349528 | 0.330152 |
| Personal → Kaggle | Palm-scale normalized | 11,680 | 24,144 | 0.400514 | 0.352665 |

## 12. Split-first production model

The first grouped fold was created before oversampling. Only personal training rows were then repeated 5×; the test fold was left untouched. The final artifact was fitted on **159,807** training rows and evaluated on **28,464** held-out rows.

- Accuracy: **0.989987**
- Macro-F1: **0.988991**
- Fit time: **489.607 seconds**
- Classes: **29**
- Features: **63**
- Saved artifact: `model/sign_language_model_combined_splitfirst.p` (**259,829,437 bytes**)
- The existing `model/sign_language_model_combined.p` (**267,146,290 bytes**) was not overwritten. `app.py` still points to that existing filename.

Raw output from the five-fold and ablation run:

```text
combined_A-Z_rows=135319 personal_rows=14600 kaggle_rows=120719 feature_count=63 folds=5 scale=Euclidean_distance_wrist_to_middle_MCP_Landmark_9
combined_raw fold=1 train_personal=11592 train_kaggle=96619 train=154579 test=27108 fit_seconds=182.219 accuracy=0.989966 macro_f1=0.988643
combined_raw fold=2 train_personal=11792 train_kaggle=96519 train=155479 test=27008 fit_seconds=209.622 accuracy=0.991225 macro_f1=0.990008
combined_raw fold=3 train_personal=11600 train_kaggle=96615 train=154615 test=27104 fit_seconds=196.812 accuracy=0.990481 macro_f1=0.989199
combined_raw fold=4 train_personal=11720 train_kaggle=96555 train=155155 test=27044 fit_seconds=186.835 accuracy=0.989794 macro_f1=0.988311
combined_raw fold=5 train_personal=11696 train_kaggle=96568 train=155048 test=27055 fit_seconds=180.551 accuracy=0.989540 macro_f1=0.988436
combined_raw mean/sample-SD: accuracy=0.990201+/-0.000668 macro_f1=0.988919+/-0.000697
combined_palm_scale fold=1 train_personal=11592 train_kaggle=96619 train=154579 test=27108 fit_seconds=164.814 accuracy=0.989265 macro_f1=0.987754
combined_palm_scale fold=2 train_personal=11792 train_kaggle=96519 train=155479 test=27008 fit_seconds=177.791 accuracy=0.990336 macro_f1=0.989123
combined_palm_scale fold=3 train_personal=11600 train_kaggle=96615 train=154615 test=27104 fit_seconds=191.826 accuracy=0.990038 macro_f1=0.988628
combined_palm_scale fold=4 train_personal=11720 train_kaggle=96555 train=155155 test=27044 fit_seconds=215.021 accuracy=0.989573 macro_f1=0.988279
combined_palm_scale fold=5 train_personal=11696 train_kaggle=96568 train=155048 test=27055 fit_seconds=217.126 accuracy=0.988098 macro_f1=0.986927
combined_palm_scale mean/sample-SD: accuracy=0.989462+/-0.000867 macro_f1=0.988142+/-0.000843
Cross-dataset evaluation uses first grouped fold from five-fold partitions; shared duplicate/mirror groups=0
kaggle_to_personal_raw: train=96575 test=2920 fit_seconds=189.795 accuracy=0.435274 macro_f1=0.392284
kaggle_to_personal_palm_scale: train=96575 test=2920 fit_seconds=156.397 accuracy=0.511986 macro_f1=0.476805
personal_to_kaggle_raw: train=11680 test=24144 fit_seconds=11.844 accuracy=0.349528 macro_f1=0.330152
personal_to_kaggle_palm_scale: train=11680 test=24144 fit_seconds=10.038 accuracy=0.400514 macro_f1=0.352665
production_splitfirst_raw: train=159807 test=28464 fit_seconds=489.607 accuracy=0.989987 macro_f1=0.988991
production_artifact=model/sign_language_model_combined_splitfirst.p bytes=259829437
```
