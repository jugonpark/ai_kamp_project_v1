# KAMP Evaluation Subsystem Design

## 현재 구조

`train_lstm_ae.py`는 KAMP 전처리, 시계열 생성, LSTM AutoEncoder 학습, 기존
`LAST_STEP_MSE + PR_INTERSECTION` 평가와 CLI 실행을 담당한다. `training_engine.py`는
동일 학습 흐름을 QThread worker로 실행하고, `gui_app.py`는 학습 상태와 기존 평가
요약을 표시한다.

## 추가 구조

- `anomaly_scoring.py`: 8개 이상점수 알고리즘 registry와 validation-normal 보정.
- `threshold_methods.py`: 6개 임계값 알고리즘 registry와 POT fallback metadata.
- `evaluation_controller.py`: 저장 모델, KAMP 데이터, prediction 메모리 cache, 평가 metrics.
- `evaluation_worker.py`: controller를 Qt worker thread에서 호출하고 plain dict signal만 전달.
- `gui_app.py`: 기존 EVALUATION 탭에 selector, 실행 버튼, 결과, 설명, 비교표를 추가.

## 데이터 흐름

`LOAD MODEL`은 저장 모델과 CSV를 읽고 기존 KAMP_REPRODUCTION과 같은 abs 전처리,
정상 Train 15,000행 scaler fit, sequence length 20, horizon 100, validation normal/anomaly
880/300 분할을 재현한다. 모델은 validation/test를 각각 한 번 predict한다. 이후
`RUN EVALUATION`과 `COMPARE ALL`은 cache된 배열에 NumPy/SciPy 후처리만 수행한다.

## Training과 Evaluation 분리

기존 `TrainingWorker`와 START/STOP/CLI는 유지한다. 새 `EvaluationWorker`는
`model.fit`, 학습 callback, 기존 output 저장 함수를 import하거나 호출하지 않는다.
평가 실패는 signal로 GUI에 전달하며 학습 thread와 상태를 공유하지 않는다.

## Cache 구조

프로세스 메모리의 단일 cache entry를 사용한다. 키는 resolve된 model/CSV 경로와 각
파일의 `st_mtime_ns`, sequence length, horizon, Train 행 수, validation 분할 수를 포함한다.
키가 같으면 model과 validation/test prediction을 재사용하고, 파일이나 데이터 설정이
바뀌면 자동으로 다시 로드한다. 디스크 prediction cache는 만들지 않는다.

## Registry 구조

각 registry entry는 내부 ID, 표시명, 설명, 계산 함수를 가진다. Score 함수는
`X`, prediction, validation-normal calibration을 받고 test/validation에 동일한 보정을
적용한다. Threshold 함수는 validation score와 label만 받고 threshold와 fallback
metadata를 반환한다. GUI는 registry를 순회하므로 새 알고리즘 추가 시 selector 코드를
수정할 필요가 없다.

## Qt worker 구조

GUI widget은 main thread에서만 갱신한다. `EvaluationWorker`는 load/evaluate/compare-all을
수행하고 progress, result, error, finished signal을 보낸다. worker 완료 slot에서는
`thread.quit()`만 요청하고 `thread.finished`에서 객체 참조를 정리한다.

## Leakage 방지

- Robust median/MAD 및 Mahalanobis mean/covariance: validation normal만 사용.
- P99/P99.5/POT: validation normal score만 사용.
- PR Intersection/Maximum F1: validation normal+anomaly와 label 사용.
- Test: 확정된 보정값과 threshold를 적용해 최종 metrics만 계산.

## 오류 처리

입력 shape, finite value, 빈 calibration 집합을 검사한다. MAD에는 epsilon을 적용하고,
Mahalanobis covariance에는 regularization과 pseudo-inverse를 사용한다. POT fitting 실패나
부족한 excess는 P99/P99.5로 fallback하며 이유를 결과에 포함한다. 모든 예외는 GUI log와
평가 상태에 표시하고 GUI process를 종료하지 않는다.

## 향후 확장 지점

새 Score/Threshold는 registry entry 추가로 확장한다. 모델 selector는 현재 고정된
`LSTM_AUTOENCODER` entry를 표시하되 향후 predictor/CNN adapter를 연결할 수 있는 위치에
둔다. 현재 범위에서는 다른 모델, window/horizon, FFT/RMS, 디스크 cache를 구현하지 않는다.
