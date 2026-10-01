# Full Training Readiness Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit and complete the CNN-LSTM experiment platform before any full training run.

**Architecture:** Keep model/data logic outside Qt, pass a complete validated configuration into `TrainingWorker`, and persist immutable run artifacts and evaluation records. The GUI remains a thread-safe view/controller over plain signal data.

**Tech Stack:** Python, TensorFlow/Keras, NumPy, pandas, PySide6, matplotlib, unittest

**Spec:** `C:/Users/jugon/.codex/attachments/b2639d03-ba96-4637-8811-8cb5dd2564fd/붙여넣은 텍스트.txt`

## Global Constraints

- Do not run 800 epoch Full Training or Full Evaluation.
- Preserve all existing model, result, metadata, CSV, and Keras files.
- Training selector exposes only CNN-LSTM and Denoising CNN-LSTM.
- Noise applies only to a copied training input; validation and test stay clean.
- Test thresholds are calibrated independently from each model's validation scores.

## Review Focus

- Invalid callback and loss values must be rejected before a worker starts.
- Manual parameter edits must switch the preset label to CUSTOM without recursive signals.
- Historical CSV imports must tolerate missing columns and reject duplicates.
- Run metadata and config JSON must retain every callback and denoising value.
- Qt widgets and matplotlib must only be updated on the GUI thread.

---

### Task 1: Training configuration and denoising contract

**Files:**
- Create: `training_config.py`
- Modify: `model_data.py`, `training_engine.py`
- Test: `tests/test_training_config.py`, `tests/test_cnn_experiment.py`

**Interfaces:**
- Produces: `default_training_config()`, `validate_training_config(config)`, `apply_random_seed(seed)` and clean/noisy training bundle behavior.

- [ ] Write failing tests for validation, seed application, clean targets, and clean validation/test.
- [ ] Implement the configuration, optimizer/loss/callback propagation, and deterministic seed setup.
- [ ] Run focused tests and confirm they pass.

### Task 2: Immutable run artifacts and evaluation records

**Files:**
- Modify: `model_artifacts.py`, `evaluation_controller.py`, `plot_utils.py`
- Test: `tests/test_model_artifacts.py`, `tests/test_evaluation_controller.py`

**Interfaces:**
- Produces: complete `metadata.json`, `experiment_summary.json`, marked training plot, and per-run `evaluation_results.csv` append.

- [ ] Write failing artifact contract tests.
- [ ] Persist callback/noise settings, best values, duration, and evaluation records.
- [ ] Run focused tests and confirm no legacy file is modified.

### Task 3: Full training controls, guide, presets, and monitor

**Files:**
- Modify: `gui_app.py`
- Test: `tests/test_gui_static.py`

**Interfaces:**
- Consumes: validated configuration from Task 1.
- Produces: full callback/denoising controls, dynamic guide, CUSTOM detection, validation, monitor and graph markers.

- [ ] Write failing offscreen GUI tests for controls and preset behavior.
- [ ] Add widgets, guide content, validation, worker config mapping, and monitor fields.
- [ ] Run offscreen GUI tests.

### Task 4: Experiment history comparison

**Files:**
- Modify: `gui_app.py`, `model_artifacts.py`
- Test: `tests/test_gui_static.py`, `tests/test_model_artifacts.py`

**Interfaces:**
- Produces: complete results columns, filters, duplicate-safe import, baseline/champion summaries, deltas, confusion matrices, and evaluation-method lock.

- [ ] Write failing tests for columns, filters, deduplication and manual baseline/champion state.
- [ ] Implement history hydration and comparison views without training.
- [ ] Run focused tests.

### Task 5: Documentation and final audit

**Files:**
- Modify: `README.txt`, `docs/cnn_experiment_platform_design.md`

- [ ] Add leakage, split, determinism, recommended experiment, and fair comparison guidance.
- [ ] Run compileall, all unit tests, and offscreen GUI construction.
- [ ] Complete the requested YES/NO verification table without running full training/evaluation.
