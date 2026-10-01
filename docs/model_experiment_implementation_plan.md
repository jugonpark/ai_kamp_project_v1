# Model Experiment Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 KAMP 경로를 보존하면서 4개 모델의 학습·저장·평가 비교를 지원한다.

**Architecture:** Model registry가 builder와 metadata를 제공하고 Task Adapter가 Reconstruction/Forecast target 차이를 감싼다. Training과 Evaluation은 공통 bundle을 소비하며 별도 QThread를 유지한다.

**Tech Stack:** Python, TensorFlow/Keras, NumPy, pandas, scikit-learn, PySide6

**Spec:** `docs/model_experiment_platform_design.md`

## Global Constraints

- 기존 KAMP builder/CLI/model/outputs를 변경하거나 덮어쓰지 않는다.
- Full Training과 Full Evaluation을 실행하지 않는다.
- sequence=20, horizon=100, feature 3개와 현재 split을 유지한다.
- Forecast target에 Equipment_state를 포함하지 않는다.

## Review Focus

- Forecast input/target/label index 정렬과 기존 sample count.
- Legacy KAMP run discovery 및 load.
- Model별 cache 격리.
- T=5 Top-K 최소 1 timestep 처리.
- Evaluation/Compare Models의 training 호출 부재.

---

### Task 1: Model Registry, Builders, Task Adapter

**Files:** Create `model_builders.py`, `model_registry.py`, `model_data.py`; Test `tests/test_model_platform.py`

**Interfaces:** `MODEL_REGISTRY`, `build_model(model_id)`, `create_task_bundle(data, model_spec)`.

- [ ] Registry 4개, output shape, KAMP builder identity, forecast alignment test를 먼저 작성한다.
- [ ] 실패를 확인하고 builder/adapter를 구현한다.
- [ ] T=20/T=5 score test와 함께 통과시킨다.

### Task 2: Artifact Storage and General Training

**Files:** Create `model_artifacts.py`; Modify `training_engine.py`; Test `tests/test_model_artifacts.py`

**Interfaces:** `save_model_run(...) -> ModelRun`, `discover_model_runs(model_id)`, `TrainingWorker(quick_test, selected_model_id)`.

- [ ] legacy/new run discovery와 metadata test를 작성한다.
- [ ] timestamp 저장과 selected model training flow를 구현한다.
- [ ] 기존 callback/STOP policy가 유지되는지 검증한다.

### Task 3: Multi-model Evaluation

**Files:** Modify `evaluation_controller.py`, `evaluation_worker.py`; Test `tests/test_evaluation_controller.py`

**Interfaces:** `EvaluationController(model_run)`, model-specific cache key, `compare_models(...)`.

- [ ] task-aware target/prediction/error와 cache key test를 작성한다.
- [ ] run load/evaluate/compare-models를 구현한다.
- [ ] Evaluation 계층에 fit 호출이 없음을 검증한다.

### Task 4: GUI and Documentation

**Files:** Modify `gui_app.py`, `README.txt`, GUI tests.

- [ ] Training/Evaluation model selector와 기본값 test를 작성한다.
- [ ] 동적 설명/architecture/run/history/Compare Models UI를 구현한다.
- [ ] compile/import, dummy forward, offscreen GUI, CLI import를 검증한다.
