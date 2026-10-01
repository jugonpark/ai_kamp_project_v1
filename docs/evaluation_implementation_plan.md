# KAMP Evaluation Subsystem Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 KAMP 학습 흐름을 보존하면서 저장된 LSTM AutoEncoder를 여러 Score/Threshold 조합으로 비교하는 GUI 평가 시스템을 추가한다.

**Architecture:** registry 기반 Score/Threshold 모듈을 순수 NumPy/SciPy 계층으로 분리하고, EvaluationController가 기존 KAMP 데이터 파이프라인과 메모리 prediction cache를 소유한다. GUI는 별도 EvaluationWorker QThread를 통해 controller를 호출한다.

**Tech Stack:** Python, NumPy, SciPy, scikit-learn, TensorFlow/Keras, PySide6

**Spec:** `docs/evaluation_subsystem_design.md`

## Global Constraints

- KAMP_REPRODUCTION, LAST_STEP_MSE, PR_INTERSECTION, 기존 GUI Training, CLI, model, outputs를 보존한다.
- FULL TRAINING, FULL EVALUATION, model.fit을 이번 검증에서 실행하지 않는다.
- Threshold와 score calibration에 Test 데이터를 사용하지 않는다.
- Prediction cache는 메모리만 사용한다.

## Review Focus

- 0 MAD channel은 epsilon scale로 finite score를 반환한다.
- singular covariance는 regularization과 pinv로 처리한다.
- POT sample 부족/fit 실패는 지정 percentile로 fallback하고 이유를 반환한다.
- model/CSV 변경은 cache key를 바꾼다.
- worker 오류는 signal로 전달되고 GUI process가 유지된다.

---

### Task 1: Score와 Threshold Registry

**Files:** Create `anomaly_scoring.py`, `threshold_methods.py`; Test `tests/test_evaluation_algorithms.py`

**Interfaces:** `fit_score_calibration(method_id, valid_normal_error)`, `compute_scores(method_id, error, calibration)`; `calculate_threshold(method_id, valid_scores, y_valid)`.

- [ ] 8개 Score와 6개 Threshold registry 및 baseline 공식 test를 작성한다.
- [ ] 실패 test를 확인한다.
- [ ] robust, top-k, time P90, Mahalanobis와 POT fallback을 구현한다.
- [ ] algorithm test를 통과시킨다.

### Task 2: EvaluationController와 Memory Cache

**Files:** Create `evaluation_controller.py`; Test `tests/test_evaluation_controller.py`

**Interfaces:** `load_model_and_predictions(force=False) -> dict`, `evaluate(score_id, threshold_id) -> dict`, `compare_all() -> list[dict]`.

- [ ] cache key, no-fit 경로, metric schema test를 작성한다.
- [ ] 기존 KAMP 전처리/분할을 재사용해 controller를 구현한다.
- [ ] validation/test predict가 cache miss에서만 실행되는지 검증한다.

### Task 3: Qt Evaluation Worker와 GUI

**Files:** Create `evaluation_worker.py`; Modify `gui_app.py`

**Interfaces:** worker signals `loaded`, `result_ready`, `comparison_ready`, `failed`, `finished`; GUI selector IDs는 registry key를 `itemData`로 보관한다.

- [ ] EVALUATION 탭에 model/score/threshold selector, 설명, metrics, confusion matrix, action buttons, history table을 추가한다.
- [ ] LOAD/RUN/COMPARE ALL을 독립 QThread에 연결하고 widget 접근은 main thread slot에서만 수행한다.
- [ ] CLEAR는 table만 비우고 EXPORT는 timestamp CSV를 새로 만든다.

### Task 4: Documentation and Static Verification

**Files:** Modify `README.txt`, `requirements.txt`; Test `tests/test_gui_static.py`

- [ ] SciPy와 새 GUI 사용법을 문서화한다.
- [ ] compile/import, registry counts, baseline selector, controller construction, offscreen instantiation을 검증한다.
- [ ] Evaluation code에 `fit(` 호출이 없고 기존 CLI와 TrainingWorker가 유지되는지 정적 검증한다.
- [ ] FULL TRAINING과 FULL EVALUATION을 실행하지 않은 상태를 보고한다.
