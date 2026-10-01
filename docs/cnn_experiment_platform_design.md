# CNN-LSTM 실험 플랫폼 설계

## 목적

기존 KAMP LSTM-AutoEncoder 기준 결과를 보존하면서 CNN-LSTM AutoEncoder와 Denoising CNN-LSTM AutoEncoder를 동일한 데이터 분할과 평가 파이프라인에서 비교한다. 실험은 명시적으로 시작해야 하며, GUI 시작 시 자동 학습하지 않는다.

## 모델 상태 정책

- `KAMP_LSTM_AE`: REFERENCE. 기존 가이드북 재현 모델과 산출물을 보존한다.
- `CNN_LSTM_AUTOENCODER`: ACTIVE. 신규 기본 실험 모델이다.
- `DENOISING_CNN_LSTM_AUTOENCODER`: ACTIVE. 학습 입력에만 Gaussian noise를 추가한다.
- GRU와 Forecast 모델: HISTORICAL. 기존 저장 run의 평가와 비교는 허용하지만 신규 LIVE TRAINING 기본 선택에서는 제외한다.

모델 run은 `outputs/models/<MODEL_ID>/<run_id>/`에 저장하며 기존 `outputs/model/` 산출물을 덮어쓰지 않는다.

## 데이터와 누수 방지

공통 전처리와 시간순 분할을 사용하고, scaler는 정상 학습 구간에만 fit한다. Validation은 threshold 선택에만 사용하며 Test는 최종 평가에만 사용한다. Denoising 모델의 noise는 `fit` 입력에만 적용하고 validation/test 입력과 label은 clean 상태로 유지한다.

## 학습 설정

GUI는 preset과 개별 설정을 함께 제공한다.

- Preset: KAMP BASELINE, CNN-LSTM BASELINE, CNN-LSTM CONSERVATIVE, DENOISING DEFAULT, CUSTOM
- 공통: epochs, batch size, learning rate, optimizer, loss, random seed
- Denoising: noise standard deviation
- `SAVE CONFIG`/`LOAD CONFIG`는 JSON으로 설정을 재현한다.

기본 학습은 800 epoch 상한과 기존 조기 종료 정책을 사용한다. QUICK_TEST는 구조와 연결을 확인하기 위한 짧은 실행이며 전체 학습 검증을 대신하지 않는다.

## 평가와 비교

저장 모델을 먼저 LOAD MODEL한 뒤 score와 threshold를 선택한다. prediction은 메모리에 cache하고 score/threshold 조합만 후처리해 비교한다. 동일한 score/threshold로 모델 간 비교할 수 있으며 결과는 GUI 표에서 CSV로 내보낸다.

Baseline과 Champion은 현재 평가 결과를 명시적으로 지정하는 포인터이며, historical CSV는 `IMPORT HISTORICAL RESULT`로 비교 표에 추가한다. 이 결과들은 신규 학습 산출물과 섞이지 않는다.

## 재현성 및 검증 경계

run metadata에는 model id/status, task, optimizer, learning rate, batch size, loss, seed, noise 설정과 데이터 분할 설명을 기록한다. 코드/정적 검증과 QUICK_TEST 결과를 FULL_TRAINING_VERIFIED로 표시하지 않는다. 전체 800 epoch 학습과 실제 운영 임계값 확정은 별도 실행과 검토가 필요하다.

Python random, NumPy, TensorFlow에 동일 seed를 적용하고 deterministic op를 요청한다. 하드웨어와 TensorFlow 실행 환경 차이 때문에 bitwise 동일성은 보장하지 않는다. Test F1을 반복적인 hyperparameter 선택 기준으로 사용하지 않으며, 현재 KAMP-compatible split 이후 STRICT_TIME_SPLIT을 별도로 검증한다.

첫 비교는 CNN-LSTM baseline과 Denoising CNN-LSTM의 noise 적용 여부만 다르게 고정한다. 양쪽 모두 MAHALANOBIS_ERROR와 POT_1PCT method를 사용하되 threshold 값은 각 모델 validation score에서 독립적으로 계산한다.
