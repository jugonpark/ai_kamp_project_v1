"""KAMP hydraulic-pump anomaly detection baseline using an LSTM AutoEncoder."""
import json
import os
import platform
import random
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("MPLBACKEND", "Agg")
# Keras 모델 저장 시 사용하는 임시 파일도 프로젝트 쓰기 가능 범위에 둔다.
RUNTIME_TEMP_DIR = Path(__file__).resolve().parent / "outputs" / ".tmp"
RUNTIME_TEMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(RUNTIME_TEMP_DIR)
os.environ["TMP"] = str(RUNTIME_TEMP_DIR)

import matplotlib.pyplot as plt
import plot_utils
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score)
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

SEED = 42
MODE = "KAMP_REPRODUCTION"
BASE_DIR = Path(__file__).resolve().parent
NORMAL_PATH = BASE_DIR / "press_data_normal.csv"
OUTLIER_PATH = BASE_DIR / "outlier_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "model"
FEATURES = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
LABEL_COLUMN = "Equipment_state"
TIMESTAMP_COLUMN = "TimeStamp"
SEQUENCE_LENGTH = 20
PREDICTION_HORIZON = 100
BATCH_SIZE = 128
EPOCHS = 800
BASELINE_EPOCHS = 200
QUICK_TEST = False  # Quick Test는 실행 전에 True로 바꾼다.
QUICK_TEST_EPOCHS = 3
QUICK_TEST_MAX_ROWS = {"train": 4000, "validation": 1000,
                       "normal_test": 1000, "anomaly_test": 600}
THRESHOLD_METHOD = "percentile"
THRESHOLD_PERCENTILE = 99
KAMP_TRAIN_ROWS = 15_000
KAMP_VALID_NORMAL_SEQUENCES = 880
KAMP_VALID_ANOMALY_SEQUENCES = 300


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def print_environment():
    gpus = tf.config.list_physical_devices("GPU")
    device = f"GPU ({len(gpus)}개)" if gpus else "CPU"
    print("=" * 60, "\nKAMP LSTM-AutoEncoder\n", "=" * 60, sep="")
    print("Python version     :", platform.python_version())
    print("Python executable  :", sys.executable)
    print("TensorFlow version :", tf.__version__)
    print("Training device    :", device)
    if not gpus:
        print("GPU가 없어 CPU로 실행합니다.")
    return device


def load_data(path):
    if not path.is_file():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {path}")
    data = pd.read_csv(path)
    data = data.loc[:, ~data.columns.astype(str).str.startswith("Unnamed")].copy()
    if data.empty:
        raise ValueError(f"CSV 파일에 데이터가 없습니다: {path}")
    return data


def validate_data(data, dataset_name, expected_label):
    required = FEATURES + [LABEL_COLUMN]
    missing_columns = [column for column in required if column not in data.columns]
    if missing_columns:
        raise ValueError(f"{dataset_name} 데이터에 필수 컬럼이 없습니다: {missing_columns}")

    data = data.copy()
    for feature in FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
    inf_count = int(np.isinf(data[FEATURES].to_numpy(dtype=float)).sum())
    if inf_count:
        print(f"[경고] {dataset_name}: inf/-inf {inf_count}개를 결측치로 처리합니다.")
        data[FEATURES] = data[FEATURES].replace([np.inf, -np.inf], np.nan)
    missing_before = data[FEATURES].isna().sum()
    if int(missing_before.sum()):
        print(f"[경고] {dataset_name}: 센서 결측치를 과거값 우선으로 보정합니다.")
        print(missing_before)
        # 이 함수는 분할 후 각 구간에 따로 적용되어 구간 간 누수를 막는다.
        data[FEATURES] = data[FEATURES].ffill().bfill()
    if data[FEATURES].isna().any().any():
        raise ValueError(f"{dataset_name}: 보정 후에도 센서 결측치가 남아 있습니다.")

    labels = pd.to_numeric(data[LABEL_COLUMN], errors="coerce")
    if labels.isna().any() or not labels.eq(expected_label).all():
        found = sorted(data[LABEL_COLUMN].dropna().astype(str).unique().tolist())
        raise ValueError(f"{dataset_name}: {LABEL_COLUMN}은 모두 {expected_label}이어야 합니다. 확인값: {found}")

    if TIMESTAMP_COLUMN in data.columns:
        timestamps = pd.to_datetime(data[TIMESTAMP_COLUMN], errors="coerce")
        if timestamps.isna().any():
            raise ValueError(f"{dataset_name}: 해석할 수 없는 TimeStamp가 있습니다.")
        if not timestamps.is_monotonic_increasing:
            raise ValueError(f"{dataset_name}: TimeStamp가 시간순으로 정렬되어 있지 않습니다.")
        duplicate_count = int(timestamps.duplicated().sum())
        if duplicate_count:
            print(f"[경고] {dataset_name}: 중복 TimeStamp {duplicate_count}개가 있습니다. 원본 행은 유지합니다.")
    if len(data) < SEQUENCE_LENGTH:
        raise ValueError(f"{dataset_name}: 데이터 길이({len(data)})가 sequence length({SEQUENCE_LENGTH})보다 짧습니다.")
    return data


def split_normal_data(normal):
    train_end = int(len(normal) * 0.70)
    valid_end = int(len(normal) * 0.85)
    splits = (normal.iloc[:train_end].copy(), normal.iloc[train_end:valid_end].copy(),
              normal.iloc[valid_end:].copy())
    for name, split in zip(("Train", "Validation", "Test"), splits):
        if len(split) < SEQUENCE_LENGTH:
            raise ValueError(f"정상 {name} 길이가 sequence length보다 짧습니다: {len(split)}")
    return splits


def apply_quick_test_subset(train, validation, normal_test, anomaly_test):
    if not QUICK_TEST:
        return train, validation, normal_test, anomaly_test
    print("\n[Quick Test] 시계열 순서를 유지한 앞부분 subset을 사용합니다.")
    return (train.iloc[:QUICK_TEST_MAX_ROWS["train"]].copy(),
            validation.iloc[:QUICK_TEST_MAX_ROWS["validation"]].copy(),
            normal_test.iloc[:QUICK_TEST_MAX_ROWS["normal_test"]].copy(),
            anomaly_test.iloc[:QUICK_TEST_MAX_ROWS["anomaly_test"]].copy())


def fit_scaler(train):
    scaler = MinMaxScaler()
    scaler.fit(train[FEATURES])
    return scaler


def transform_features(data, scaler):
    result = data.copy()
    result[FEATURES] = scaler.transform(result[FEATURES])
    return result


def create_sequences(data, sequence_length=SEQUENCE_LENGTH):
    values = data[FEATURES].to_numpy(dtype=np.float32)
    if len(values) < sequence_length:
        raise ValueError(f"데이터 길이({len(values)})가 sequence length({sequence_length})보다 짧습니다.")
    return np.stack([values[start:start + sequence_length]
                     for start in range(len(values) - sequence_length + 1)]).astype(np.float32)


def build_model(sequence_length=SEQUENCE_LENGTH, feature_count=len(FEATURES)):
    inputs = layers.Input(shape=(sequence_length, feature_count))
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.LSTM(32, return_sequences=False, name="latent_vector")(x)
    x = layers.RepeatVector(sequence_length)(x)
    x = layers.LSTM(32, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    outputs = layers.TimeDistributed(layers.Dense(feature_count))(x)
    model = Model(inputs, outputs, name="kamp_lstm_autoencoder")
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def train_model(model, x_train, x_validation):
    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=10, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1),
    ]
    return model.fit(x_train, x_train,
                     validation_data=(x_validation, x_validation),
                     epochs=QUICK_TEST_EPOCHS if QUICK_TEST else (EPOCHS if MODE == "KAMP_REPRODUCTION" else BASELINE_EPOCHS),
                     batch_size=BATCH_SIZE, shuffle=False,
                     callbacks=callbacks, verbose=1)


def calculate_reconstruction_error(model, sequences):
    predictions = model.predict(sequences, verbose=0)
    return np.mean(np.square(sequences - predictions), axis=(1, 2))


def calculate_threshold(validation_error):
    if THRESHOLD_METHOD == "percentile":
        return float(np.percentile(validation_error, THRESHOLD_PERCENTILE))
    raise ValueError(f"지원하지 않는 threshold 방식: {THRESHOLD_METHOD}. 현재는 percentile만 지원합니다.")


def evaluate_model(normal_error, anomaly_error, threshold):
    y_true = np.concatenate([np.zeros(len(normal_error)), np.ones(len(anomaly_error))]).astype(int)
    errors = np.concatenate([normal_error, anomaly_error])
    y_pred = (errors > threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["Normal", "Anomaly"], zero_division=0, output_dict=True),
    }
    return y_true, errors, y_pred, metrics


def save_results(model, history, y_true, errors, y_pred, threshold,
                 normal_error, anomaly_error, metrics, split_sizes, device):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR / "kamp_lstm_autoencoder.keras")
    pd.DataFrame(history.history).to_csv(OUTPUT_DIR / "training_history.csv", index_label="epoch")

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("LSTM AutoEncoder Training Loss")
    plt.legend(); plt.tight_layout(); plt.savefig(OUTPUT_DIR / "training_loss.png", dpi=150); plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(errors, label="Reconstruction Error")
    plt.axhline(y=threshold, linestyle="--", label="Threshold")
    plt.axvline(x=len(normal_error), linestyle=":", label="Normal / Anomaly boundary")
    plt.xlabel("Sequence"); plt.ylabel("MSE"); plt.title("KAMP Hydraulic Pump Anomaly Detection")
    plt.legend(); plt.tight_layout(); plt.savefig(OUTPUT_DIR / "reconstruction_error.png", dpi=150); plt.close()

    pd.DataFrame({"true_label": y_true, "reconstruction_error": errors,
                  "predicted_label": y_pred}).to_csv(
        OUTPUT_DIR / "anomaly_detection_results.csv", index=False, encoding="utf-8-sig")
    payload = {"threshold_method": THRESHOLD_METHOD,
               "threshold_percentile": THRESHOLD_PERCENTILE,
               "threshold": threshold,
               "normal_reconstruction_error_mean": float(normal_error.mean()),
               "anomaly_reconstruction_error_mean": float(anomaly_error.mean()), **metrics}
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["KAMP LSTM-AutoEncoder 실행 요약",
             f"실행 환경: {platform.platform()}", f"Python 버전: {platform.python_version()}",
             f"TensorFlow 버전: {tf.__version__}", f"CPU/GPU 여부: {device}",
             f"Quick Test: {QUICK_TEST}", f"사용 feature: {', '.join(FEATURES)}",
             f"Sequence Length: {SEQUENCE_LENGTH}",
             f"Train 크기: {split_sizes['train']}",
             f"Validation 크기: {split_sizes['validation']}",
             f"Normal Test 크기: {split_sizes['normal_test']}",
             f"Anomaly Test 크기: {split_sizes['anomaly_test']}",
             f"실제 학습 epoch 수: {len(history.history['loss'])}",
             f"Threshold: {threshold:.10g}",
             f"정상 Reconstruction Error 평균: {normal_error.mean():.10g}",
             f"이상 Reconstruction Error 평균: {anomaly_error.mean():.10g}",
             f"Accuracy: {metrics['accuracy']:.6f}", f"Precision: {metrics['precision']:.6f}",
             f"Recall: {metrics['recall']:.6f}", f"F1-score: {metrics['f1_score']:.6f}"]
    (OUTPUT_DIR / "run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_data_diagnostics(data, name):
    print(f"\n[{name} 데이터 진단]")
    print("shape:", data.shape)
    print("columns:", list(data.columns))
    print("dtype:\n", data.dtypes)
    print("missing:\n", data.isna().sum())
    numeric = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    print("inf/-inf:", int(np.isinf(numeric.to_numpy(dtype=float)).sum()))
    print("duplicate rows:", int(data.duplicated().sum()))
    if TIMESTAMP_COLUMN in data.columns:
        timestamps = pd.to_datetime(data[TIMESTAMP_COLUMN], errors="coerce")
        print("duplicate TimeStamp:", int(timestamps.duplicated().sum()))
    print("Equipment_state distribution:", data[LABEL_COLUMN].value_counts(dropna=False).to_dict())


def create_horizon_sequences(data, labels, sequence_length=SEQUENCE_LENGTH,
                             prediction_horizon=PREDICTION_HORIZON):
    values = data[FEATURES].to_numpy(dtype=np.float32)
    labels = np.asarray(labels, dtype=int)
    limit = len(values) - sequence_length - prediction_horizon
    if limit <= 0:
        raise ValueError("sequence length와 prediction horizon을 만들 데이터가 부족합니다.")
    x = np.stack([values[i:i + sequence_length] for i in range(limit)]).astype(np.float32)
    y = np.asarray([labels[i + sequence_length + prediction_horizon] for i in range(limit)], dtype=int)
    return x, y


def last_timestep_error(model, sequences):
    predictions = model.predict(sequences, verbose=0)
    return np.mean(np.square(sequences[:, -1, :] - predictions[:, -1, :]), axis=1)


def calculate_pr_intersection_threshold(y_valid, valid_error):
    precision, recall, thresholds = precision_recall_curve(y_valid, valid_error)
    if len(thresholds) == 0:
        raise ValueError("PR threshold를 계산할 validation error가 부족합니다.")
    index = int(np.argmin(np.abs(precision[:-1] - recall[:-1])))
    return float(thresholds[index]), float(precision[index]), float(recall[index]), precision, recall, thresholds


def save_reproduction_results(model, history, x_test, y_test, test_error, predictions,
                              threshold, valid_precision, valid_recall, metrics,
                              shapes, reference, save_static_plots=True):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR / "kamp_lstm_autoencoder.keras")
    pd.DataFrame(history.history).to_csv(OUTPUT_DIR / "training_history.csv", index_label="epoch")

    precision, recall, thresholds = precision_recall_curve(reference["y_valid"], reference["valid_error"])
    if save_static_plots:
        plot_utils.save_training_loss(history.history["loss"], history.history["val_loss"], OUTPUT_DIR / "training_loss.png")
        plot_utils.save_pr_threshold_curve(precision, recall, thresholds, threshold, OUTPUT_DIR / "pr_threshold_curve.png")
        plot_utils.save_reconstruction_error(test_error, y_test, threshold, OUTPUT_DIR / "reconstruction_error_test.png")
        plot_utils.save_confusion_matrix(metrics["confusion_matrix"], OUTPUT_DIR / "confusion_matrix.png")

    pd.DataFrame({"sample_index": np.arange(len(y_test)), "true_label": y_test,
                  "reconstruction_error": test_error, "threshold": threshold,
                  "predicted_label": predictions}).to_csv(
        OUTPUT_DIR / "anomaly_detection_results.csv", index=False, encoding="utf-8-sig")
    payload = {"mode": MODE, "sequence_length": SEQUENCE_LENGTH,
               "prediction_horizon": PREDICTION_HORIZON,
               "threshold_method": "precision_recall_intersection", "threshold": threshold,
               "validation_precision": valid_precision, "validation_recall": valid_recall,
               "accuracy": metrics["accuracy"], "precision": metrics["precision"],
               "recall": metrics["recall"], "f1": metrics["f1_score"],
               "confusion_matrix": metrics["confusion_matrix"], "shapes": shapes,
               "quick_test": QUICK_TEST}
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["Run Mode: KAMP_REPRODUCTION", f"QUICK_TEST: {QUICK_TEST}",
             f"Python version: {platform.python_version()}", f"TensorFlow version: {tf.__version__}",
             f"CPU/GPU: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}",
             "정상 데이터 행 수: 20000", "이상 데이터 행 수: 600",
             f"Features: {', '.join(FEATURES)}", "절댓값 전처리: 적용",
             f"Train 원본 길이: {KAMP_TRAIN_ROWS}", f"Sequence Length: {SEQUENCE_LENGTH}",
             f"Prediction Horizon: {PREDICTION_HORIZON}", f"X_train shape: {shapes['x_train']}",
             f"Validation Normal 개수: {shapes['valid_normal']}", f"Validation Anomaly 개수: {shapes['valid_anomaly']}",
             f"Test Normal 개수: {shapes['test_normal']}", f"Test Anomaly 개수: {shapes['test_anomaly']}",
             f"Batch Size: {BATCH_SIZE}", f"최대 Epoch: {EPOCHS}",
             f"실제 학습 Epoch: {len(history.history['loss'])}",
             "Threshold 방법: precision_recall_intersection", f"Threshold: {threshold:.10g}",
             f"Validation Precision: {valid_precision:.6f}", f"Validation Recall: {valid_recall:.6f}",
             f"Test Accuracy: {metrics['accuracy']:.6f}", f"Test Precision: {metrics['precision']:.6f}",
             f"Test Recall: {metrics['recall']:.6f}", f"Test F1: {metrics['f1_score']:.6f}",
             f"Confusion Matrix: {metrics['confusion_matrix']}",
             "KAMP Guidebook Reference: Accuracy 약 97.51%, F1 약 74.76%, 결과는 강제 비교하지 않음"]
    (OUTPUT_DIR / "run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_kamp_reproduction():
    normal = load_data(NORMAL_PATH)
    anomaly = load_data(OUTLIER_PATH)
    print_data_diagnostics(normal, "정상")
    print_data_diagnostics(anomaly, "이상")
    normal = validate_data(normal, "정상", 0)
    anomaly = validate_data(anomaly, "이상", 1)
    normal[FEATURES] = normal[FEATURES].abs()
    anomaly[FEATURES] = anomaly[FEATURES].abs()
    print("\n[KAMP abs 전처리 후 기술통계]")
    print(pd.concat([normal[FEATURES].describe().loc[["mean", "std", "min", "max"]].assign(dataset="normal"),
                     anomaly[FEATURES].describe().loc[["mean", "std", "min", "max"]].assign(dataset="anomaly")]))
    normal_train = normal.iloc[:KAMP_TRAIN_ROWS].copy()
    normal_test = normal.iloc[KAMP_TRAIN_ROWS:].copy()
    scaler = fit_scaler(normal_train)
    normal_train = transform_features(normal_train, scaler)
    normal_test = transform_features(normal_test, scaler)
    anomaly = transform_features(anomaly, scaler)
    x_train, y_train = create_horizon_sequences(normal_train, normal_train[LABEL_COLUMN])
    x_test_normal, y_test_normal = create_horizon_sequences(normal_test, normal_test[LABEL_COLUMN])
    x_test_anomaly, y_test_anomaly = create_horizon_sequences(anomaly, anomaly[LABEL_COLUMN])
    x_valid_normal, y_valid_normal = x_test_normal[:KAMP_VALID_NORMAL_SEQUENCES], y_test_normal[:KAMP_VALID_NORMAL_SEQUENCES]
    x_test_normal, y_test_normal = x_test_normal[KAMP_VALID_NORMAL_SEQUENCES:], y_test_normal[KAMP_VALID_NORMAL_SEQUENCES:]
    x_valid_anomaly, y_valid_anomaly = x_test_anomaly[:KAMP_VALID_ANOMALY_SEQUENCES], y_test_anomaly[:KAMP_VALID_ANOMALY_SEQUENCES]
    x_test_anomaly, y_test_anomaly = x_test_anomaly[KAMP_VALID_ANOMALY_SEQUENCES:], y_test_anomaly[KAMP_VALID_ANOMALY_SEQUENCES:]
    x_valid = np.vstack([x_valid_normal, x_valid_anomaly])
    y_valid = np.hstack([y_valid_normal, y_valid_anomaly])
    x_test = np.vstack([x_test_normal, x_test_anomaly])
    y_test = np.hstack([y_test_normal, y_test_anomaly])
    print("\n[KAMP Sequence Shape]")
    for name, value in (("X_train", x_train), ("X_valid_normal", x_valid_normal), ("X_valid_anomaly", x_valid_anomaly),
                        ("X_test_normal", x_test_normal), ("X_test_anomaly", x_test_anomaly), ("X_valid", x_valid), ("X_test", x_test)):
        print(name, value.shape)
    if QUICK_TEST:
        print("QUICK TEST SUBSET NOT USED: full KAMP Train sequence set is retained; only epochs are limited to 3.")
    model = build_model(); print("\n[Model]"); model.summary()
    history = train_model(model, x_train, x_valid[y_valid == 0])
    valid_error = last_timestep_error(model, x_valid)
    threshold, valid_precision, valid_recall, _, _, _ = calculate_pr_intersection_threshold(y_valid, valid_error)
    test_error = last_timestep_error(model, x_test)
    y_pred = (test_error > threshold).astype(int)
    metrics = {"accuracy": float(accuracy_score(y_test, y_pred)), "precision": float(precision_score(y_test, y_pred, zero_division=0)),
               "recall": float(recall_score(y_test, y_pred, zero_division=0)), "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
               "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()}
    print(f"\nThreshold: {threshold}\nValidation Precision: {valid_precision:.6f}\nValidation Recall: {valid_recall:.6f}")
    print("Test Normal:", int((y_test == 0).sum()), "정상→정상:", int(((y_test == 0) & (y_pred == 0)).sum()), "정상→이상:", int(((y_test == 0) & (y_pred == 1)).sum()))
    print("Test Anomaly:", int((y_test == 1).sum()), "이상→이상:", int(((y_test == 1) & (y_pred == 1)).sum()), "이상→정상:", int(((y_test == 1) & (y_pred == 0)).sum()))
    print("Metrics:", metrics)
    save_reproduction_results(model, history, x_test, y_test, test_error, y_pred, threshold, valid_precision, valid_recall, metrics,
                              {"x_train": list(x_train.shape), "valid_normal": len(x_valid_normal), "valid_anomaly": len(x_valid_anomaly),
                               "test_normal": len(x_test_normal), "test_anomaly": len(x_test_anomaly)},
                              {"y_valid": y_valid, "valid_error": valid_error})


def run_baseline():
    set_random_seed(SEED)
    device = print_environment()
    normal_raw = load_data(NORMAL_PATH)
    anomaly_raw = load_data(OUTLIER_PATH)
    # 정상 데이터는 먼저 시간순 분할하고 각 구간을 따로 보정한다.
    train, validation, normal_test = split_normal_data(normal_raw)
    train = validate_data(train, "정상 Train", 0)
    validation = validate_data(validation, "정상 Validation", 0)
    normal_test = validate_data(normal_test, "정상 Test", 0)
    anomaly = validate_data(anomaly_raw, "이상 Test", 1)
    train, validation, normal_test, anomaly = apply_quick_test_subset(
        train, validation, normal_test, anomaly)
    print("\n[행 단위 데이터 분할]")
    for name, data in (("Train Normal", train), ("Valid Normal", validation),
                       ("Test Normal", normal_test), ("Test Anomaly", anomaly)):
        print(f"{name:13}: {len(data)}")

    scaler = fit_scaler(train)  # 정상 Train에만 fit
    train, validation, normal_test, anomaly = [
        transform_features(data, scaler) for data in (train, validation, normal_test, anomaly)]
    sequences = {"train": create_sequences(train),
                 "validation": create_sequences(validation),
                 "normal_test": create_sequences(normal_test),
                 "anomaly_test": create_sequences(anomaly)}
    print("\n[Sequence Shape]")
    for name, values in sequences.items():
        print(f"{name:13}: {values.shape}")
        if values.shape[1:] != (SEQUENCE_LENGTH, len(FEATURES)):
            raise RuntimeError(f"예상하지 못한 sequence shape: {name}={values.shape}")

    model = build_model(); print("\n[Model]"); model.summary()
    history = train_model(model, sequences["train"], sequences["validation"])
    valid_error = calculate_reconstruction_error(model, sequences["validation"])
    threshold = calculate_threshold(valid_error)
    normal_error = calculate_reconstruction_error(model, sequences["normal_test"])
    anomaly_error = calculate_reconstruction_error(model, sequences["anomaly_test"])
    y_true, errors, y_pred, metrics = evaluate_model(normal_error, anomaly_error, threshold)
    print("\n" + "=" * 60)
    print(f"Threshold ({THRESHOLD_METHOD}, {THRESHOLD_PERCENTILE}%): {threshold}")
    print(f"정상 평균 Reconstruction Error: {normal_error.mean()}")
    print(f"이상 평균 Reconstruction Error: {anomaly_error.mean()}")
    for name in ("accuracy", "precision", "recall", "f1_score"):
        print(f"{name:10}: {metrics[name]:.4f}")
    print("\nConfusion Matrix\n", np.asarray(metrics["confusion_matrix"]))
    print("\nClassification Report\n", classification_report(
        y_true, y_pred, target_names=["Normal", "Anomaly"], zero_division=0))
    save_results(model, history, y_true, errors, y_pred, threshold, normal_error,
                 anomaly_error, metrics,
                 {name: len(values) for name, values in sequences.items()}, device)
    print(f"\n저장 완료: {OUTPUT_DIR}")


def main():
    if MODE == "KAMP_REPRODUCTION":
        run_kamp_reproduction()
    elif MODE == "BASELINE":
        run_baseline()
    else:
        raise ValueError(f"지원하지 않는 MODE: {MODE}. KAMP_REPRODUCTION 또는 BASELINE을 사용하세요.")


if __name__ == "__main__":
    main()
