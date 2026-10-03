"""Model metadata registry; KAMP baseline delegates to the legacy builder."""
from dataclasses import dataclass
from typing import Callable

import train_lstm_ae as core
from model_builders import (build_gru_autoencoder, build_cnn_lstm_autoencoder,
                            build_denoising_cnn_lstm_autoencoder, build_lstm_forecast_5)

ALL_SCORES = ("LAST_STEP_MSE","FULL_WINDOW_MSE","ROBUST_CHANNEL_FULL","ROBUST_TOPK_10",
              "ROBUST_TOPK_20","ROBUST_TOPK_30","TIME_P90","MAHALANOBIS_ERROR")

@dataclass(frozen=True)
class ModelSpec:
    id: str; display_name: str; task_type: str; description: str; architecture: str
    builder: Callable; forecast_length: int = 0; compatible_score_methods: tuple = ALL_SCORES
    status: str = "HISTORICAL"; denoising: bool = False

MODEL_REGISTRY = {
 "KAMP_LSTM_AE": ModelSpec("KAMP_LSTM_AE","KAMP LSTM AutoEncoder","RECONSTRUCTION","기존 KAMP 가이드북 기준 20×3 복원 모델.","INPUT 20×3\n↓\nLSTM64 → LSTM32\n↓\nRepeatVector20\n↓\nLSTM32 → LSTM64\n↓\nOUTPUT 20×3",core.build_model, status="REFERENCE"),
 "GRU_AUTOENCODER": ModelSpec("GRU_AUTOENCODER","GRU AutoEncoder","RECONSTRUCTION","GRU로 동일한 reconstruction task를 수행.","INPUT 20×3\n↓\nGRU64 → GRU32\n↓\nRepeatVector20\n↓\nGRU32 → GRU64\n↓\nOUTPUT 20×3",build_gru_autoencoder, status="HISTORICAL"),
 "CNN_LSTM_AUTOENCODER": ModelSpec("CNN_LSTM_AUTOENCODER","CNN-LSTM AutoEncoder","RECONSTRUCTION","Conv1D로 짧은 시간 구간의 진동·전류 특징을 추출하고, LSTM으로 시간적 흐름을 학습한 뒤 정상 센서 패턴을 복원하는 모델입니다.","입력 20×3\n↓\nConv1D32 × 2\n↓\nLSTM64 → LSTM32\n↓\nRepeatVector20\n↓\nLSTM32 → LSTM64\n↓\n출력 20×3",build_cnn_lstm_autoencoder, status="ACTIVE"),
 "DENOISING_CNN_LSTM_AUTOENCODER": ModelSpec("DENOISING_CNN_LSTM_AUTOENCODER","Denoising CNN-LSTM AutoEncoder","RECONSTRUCTION","학습 중 정상 데이터에 매번 새로운 Gaussian 노이즈를 추가하고 원래의 깨끗한 정상 데이터를 복원하도록 학습합니다. 특정 노이즈 패턴 하나를 외우지 않고 다양한 작은 센서 흔들림에 강건한 정상 패턴을 학습하는 것이 목적입니다. 검증 및 테스트 데이터에는 노이즈를 추가하지 않습니다.","입력 20×3 + Gaussian 노이즈\n↓\nConv1D32 × 2\n↓\nLSTM64 → LSTM32\n↓\nRepeatVector20\n↓\nLSTM32 → LSTM64\n↓\n출력 20×3",build_denoising_cnn_lstm_autoencoder, status="ACTIVE", denoising=True),
 "LSTM_FORECAST_5": ModelSpec("LSTM_FORECAST_5","LSTM Forecast 5-Step","FORECAST","과거 20 timestep으로 다음 5 timestep 센서를 예측.","INPUT 20×3\n↓\nLSTM64 → LSTM32 → Dense64\n↓\nRepeatVector5\n↓\nLSTM32 → LSTM64\n↓\nFORECAST 5×3",build_lstm_forecast_5,5, status="HISTORICAL"),
}

PROCESSED_DATASET_MODEL_IDS = (
    "KAMP_LSTM_AE", "CNN_LSTM_AUTOENCODER", "DENOISING_CNN_LSTM_AUTOENCODER",
)

def build_model(model_id, sequence_length=None, feature_count=None):
    spec = MODEL_REGISTRY[model_id]
    if spec.task_type != "RECONSTRUCTION":
        if sequence_length is not None or feature_count is not None:
            raise ValueError("Dynamic sequence is only supported for reconstruction models")
        return spec.builder()
    kwargs = {}
    if sequence_length is not None: kwargs["sequence_length"] = sequence_length
    if feature_count is not None: kwargs["feature_count"] = feature_count
    return spec.builder(**kwargs)
