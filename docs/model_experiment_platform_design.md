# Model Experiment Platform Design

## 목적과 보존 경계

GUI를 `MODEL -> ANOMALY SCORE -> THRESHOLD` 실험 플랫폼으로 확장한다. 기존
`train_lstm_ae.py`의 KAMP_REPRODUCTION, builder, CLI, legacy model 및 outputs는
수정하거나 이동하지 않는다. GUI 학습 결과는 새 run 디렉터리에만 저장한다.

## Model Registry와 Builder

`model_registry.py`는 ID, 표시명, task type, 설명, architecture text, builder,
forecast length, compatible score를 보관한다. `KAMP_LSTM_AE` builder는 기존
`train_lstm_ae.build_model()`을 직접 호출한다. 나머지 builder는 `model_builders.py`에
구현하며 모두 Adam(0.001), MSE로 compile한다.

## Task Adapter와 Data Bundle

`model_data.py`의 adapter는 모델별 분기 대신 task 차이를 캡슐화한다. 공통 bundle은
`inputs`, `targets`, `labels`, `task_type`, `error_name`을 가진다.

- Reconstruction: input `i:i+20`, target=input, label `i+20+100`.
- Forecast: input `i:i+20`, target `i+20:i+25`, label `i+20+100`.

샘플 수는 기존 `len(data)-20-100`과 동일하다. Equipment_state는 label에만 쓰인다.

## Training Flow

TrainingWorker는 selected model ID로 registry와 adapter를 선택한다. 공통 callback과
policy를 유지하고 `model.fit(bundle.inputs, bundle.targets)`를 실행한다. 완료 후
`model_artifacts.py`가 새 timestamp run에 model, metadata, history CSV, loss PNG를 저장한다.
STOP은 기존 callback의 `model.stop_training=True`를 유지한다.

## Evaluation Flow

EvaluationController는 선택된 ModelRun을 로드하고 같은 adapter로 validation/test target을
생성한다. prediction과 target의 제곱 오차를 score registry에 전달한다. Threshold는 해당
모델의 validation score로 매번 보정하며 test는 최종 metrics에만 사용한다.

## Artifact와 Run Discovery

새 run은 `outputs/models/<MODEL_ID>/<timestamp>/`에 저장한다. legacy
`outputs/model/kamp_lstm_autoencoder.keras`는 KAMP_LSTM_AE의 `LEGACY_KAMP` run으로
노출한다. metadata가 없는 legacy run은 registry 기본 metadata를 사용한다.

## Cache

키는 model ID/task/forecast length, model 경로·mtime·size, CSV 상태, sequence/horizon,
feature/split 설정을 포함한다. 서로 다른 모델의 prediction은 공유하지 않는다.

## Score 일반화

Error 입력은 `(samples,T,3)`, `T>=1`이다. Top-K는 `max(1, ceil(T*ratio))`를 사용한다.
Reconstruction은 Reconstruction Error, Forecast는 Prediction Error로 표시한다.

## GUI Flow

LIVE TRAINING은 Training Model selector, 설명, 동적 architecture를 제공한다. EVALUATION은
Model Type/Run, Score, Threshold와 LOAD/RUN/COMPARE ALL/COMPARE MODELS를 제공한다.
기본 조합은 KAMP_LSTM_AE + LAST_STEP_MSE + PR_INTERSECTION이다.

## 확장

새 모델은 builder와 registry entry, 필요한 task adapter만 추가한다. 현재 범위에는
hyperparameter tuning, strict time split, FFT/RMS, Transformer/TCN, disk prediction cache가 없다.
