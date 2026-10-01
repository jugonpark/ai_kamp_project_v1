KAMP LSTM-AutoEncoder Baseline / Guidebook Reproduction
=======================================================

다른 Windows PC에서 시작하기
----------------------------
Windows 10/11 64-bit와 Python 3.12 64-bit를 권장합니다. 현재 설치 조합은
Python 3.13 64-bit에서도 검증했습니다. TensorFlow는 CPU 실행이 기본이며
CUDA나 GPU 설치는 필요하지 않습니다.

1. GitHub repository를 clone하거나 ZIP으로 다운로드해 압축을 풉니다.
2. Python 3.12 64-bit를 설치합니다. 설치 시 PATH 추가를 선택하면 편리합니다.
3. 프로젝트 폴더의 `setup_environment.bat`를 더블클릭합니다. `.venv` 생성,
   pip 및 `requirements.txt` 설치, 의존성 검증이 차례로 진행됩니다.
4. VS Code를 쓴다면 Python Interpreter로 `.venv\Scripts\python.exe`를 선택합니다.
5. `run_gui.bat`로 GUI를 엽니다. 오류가 나면 `run_gui_debug.bat`로 실행해
   콘솔의 traceback을 확인합니다.

프로젝트가 OneDrive, 공백 또는 한글이 포함된 경로에 있어도 배치 파일은 자신의
위치를 기준으로 실행합니다. 시스템 Python에 직접 패키지를 설치하지 않습니다.
환경 확인 결과는 `outputs/environment/environment_*.txt`에 저장됩니다.

환경 문제 해결
--------------
- `ModuleNotFoundError: No module named 'PySide6'`: 시스템 Python으로 실행했거나
  설치가 끝나지 않은 상태입니다. `setup_environment.bat`를 다시 실행하고 VS Code의
  interpreter를 프로젝트 `.venv\Scripts\python.exe`로 선택하세요.
- `TensorFlow GPU support is not available`: Windows CPU 환경에서는 정상일 수
  있습니다. `verify_environment.py`가 CPU device를 확인하면 GUI를 사용할 수 있습니다.
- GUI가 바로 종료된다면 `run_gui_debug.bat`에서 오류 내용을 확인하세요.
- 기존 `.venv`의 Python 버전이 맞지 않으면 해당 가상환경을 별도로 백업한 뒤
  Python 3.12/3.13 64-bit로 다시 생성하세요.

Desktop GUI
-----------
PySide6 기반 관제 화면은 다음 명령으로 실행합니다.

    run_gui.bat

`run_gui.bat`를 더블클릭해도 됩니다. 프로젝트의 `.venv\Scripts\pythonw.exe`를
사용해 콘솔 창 없이 GUI를 엽니다.

    .\.venv\Scripts\python.exe gui_app.py

GUI는 시작 시 학습하지 않습니다. `Run Mode`에서 `QUICK_TEST` 또는
`FULL_TRAINING`을 고르고 `START TRAINING`을 눌러야 worker thread에서 학습이
시작됩니다. `STOP`은 epoch 경계에서 안전하게 중지하도록 요청하며,
`OPEN OUTPUTS`는 프로젝트의 outputs 폴더를 엽니다.

LIVE TRAINING의 왼쪽은 스크롤 가능한 설정 영역입니다. MODEL SETTINGS 아래
TRAINING PARAMETERS에서 Epoch, Batch Size, Learning Rate, Optimizer, Loss,
Random Seed를 수정하고 CALLBACK SETTINGS와 DENOISING SETTINGS를 확인합니다.
화면 하단의 Current Experiment Summary와 START/STOP은 스크롤 위치와 관계없이
표시됩니다. 좌우 폭은 구분선을 드래그해 조정할 수 있습니다.
학습 중에는 모델, 파라미터, preset, 설정 불러오기와 초기화가 잠기며 종료 후 다시
활성화됩니다. 숫자는 입력칸에 직접 입력하고 선택 항목은 콤보박스를 열어서
변경합니다. 마우스 휠은 값을 바꾸지 않고 설정 영역을 스크롤합니다. 각 항목의
`ⓘ` 버튼을 누르면 현재 값, 의미와 권장 범위가 표시됩니다.

`학습 그래프` 탭에서는 현재 학습과 저장된 실행의 `training_history.csv`를 큰
화면으로 볼 수 있습니다. 손실·학습률·일반화 차이·손실과 학습률의 분리 보기,
두 실행의 손실 비교를 지원합니다. Matplotlib 도구 모음으로 확대·이동·초기화할
수 있으며 `그래프 PNG 저장`은 `outputs/graphs/`에 새 파일을 생성합니다.
실시간 학습의 `학습 그래프 크게 보기`와 평가 및 비교의 `학습 곡선 보기`로
해당 탭에 이동할 수 있습니다. 그래프 조회는 학습이나 평가를 시작하지 않습니다.

저장 모델 평가와 알고리즘 비교
-----------------------------
`EVALUATION` 탭은 기존 Training과 분리되어 있으며 평가 동작에서 모델을 다시
학습하지 않습니다. 기본 선택은 기존 KAMP 기준인 `LAST_STEP_MSE`와
`PR_INTERSECTION`입니다.

    .\.venv\Scripts\python.exe gui_app.py

실행 후 `EVALUATION` -> `LOAD MODEL`을 누르면 저장된 모델과 기존 KAMP 데이터
분할로 Validation/Test prediction을 각각 한 번 생성해 메모리에 보관합니다. 이후
Score와 Threshold를 선택하고 `RUN EVALUATION`을 누르면 cache된 prediction에
후처리만 적용합니다. `COMPARE ALL`은 8 x 6 조합을 같은 cache로 비교합니다.

지원 Score: LAST_STEP_MSE, FULL_WINDOW_MSE, ROBUST_CHANNEL_FULL,
ROBUST_TOPK_10/20/30, TIME_P90, MAHALANOBIS_ERROR.

지원 Threshold: PR_INTERSECTION, MAX_F1, NORMAL_P99, NORMAL_P995,
POT_1PCT, POT_0P5PCT. POT 적합이 불가능하면 지정 percentile 방식으로 fallback하고
GUI 결과와 log에 이유를 표시합니다.

`CLEAR COMPARISON`은 현재 GUI 표만 비웁니다. `EXPORT COMPARISON CSV`는 사용자가
누를 때만 `outputs/gui_comparisons/comparison_<timestamp>.csv`를 새로 생성합니다.

모델 실험 플랫폼
----------------
 LIVE TRAINING의 `TRAINING MODEL`에서 ACTIVE 상태인 CNN-LSTM AutoEncoder와
 Denoising CNN-LSTM AutoEncoder를 신규 학습 모델로 선택할 수 있습니다. KAMP
 LSTM-AutoEncoder는 REFERENCE로 평가/재현에 보존되고, GRU와 Forecast 모델은
 HISTORICAL run 평가에 사용할 수 있습니다. GUI 학습 결과는
기존 legacy 모델을 덮어쓰지 않고 다음 위치에 run별로 저장됩니다.

    outputs/models/<MODEL_ID>/<timestamp>/

EVALUATION에서는 `MODEL TYPE`과 `MODEL RUN`을 선택한 뒤 LOAD MODEL을 실행합니다.
`COMPARE MODELS`는 현재 Score/Threshold를 고정하고 각 모델의 최신 저장 run을
평가합니다. 학습을 실행하지 않습니다.

`TRAINING CONFIGURATION`의 preset, hyperparameter, random seed, denoising noise
설정을 조정할 수 있습니다. `SAVE CONFIG`와 `LOAD CONFIG`는 JSON 설정을
`configs/`에 저장해 같은 실험 조건을 재현합니다. Denoising noise는 training
input에만 적용되고 validation/test는 clean 상태로 평가합니다.

ReduceLROnPlateau와 EarlyStopping의 활성화, factor, patience, minimum learning
rate, min delta, best weight 복원을 GUI에서 설정할 수 있습니다. AdamW의 weight
decay와 Huber delta도 실제 optimizer/loss에 전달됩니다. Python random, NumPy,
TensorFlow seed와 deterministic op 설정을 적용하지만 플랫폼에 따른 완전한 bitwise
재현성까지 보장하지는 않습니다.

첫 Denoising 비교는 CNN-LSTM baseline과 seed, optimizer, learning rate, batch,
 loss, callback을 같게 두고 Gaussian noise std 0.005부터 비교합니다. 평가는
MAHALANOBIS_ERROR + POT_1PCT를 우선 사용하되, threshold 숫자는 각 모델의
validation score로 따로 보정합니다. Test metric을 반복적으로 hyperparameter 선택에
사용하면 test set에 간접 과적합될 수 있으므로 최종 확인 용도로만 사용합니다.

새 Denoising 학습은 `tf.data`에서 각 배치를 소비할 때마다 Gaussian 노이즈를 다시
생성합니다. 입력은 noisy, target은 clean이며 검증 및 테스트 데이터에는 노이즈를
추가하지 않습니다. `DENOISING WEAK 0.005` 프리셋은 첫 약한 노이즈 비교용입니다.
기존 저장 모델의 메타데이터에 `noise_generation`이 없으면 로드 시
`legacy_static`으로 해석하고, 새 실행은 `dynamic_per_batch`로 기록합니다.

Reconstruction 모델은 입력 20×3을 복원하고 Forecast 모델은 같은 입력에서 바로 다음
5×3 센서값을 예측합니다. 두 task의 label은 기존과 동일한 input 시작점 기준
`index + 20 + 100` 위치입니다. 현재 split은 KAMP Guidebook-compatible 방식이며 향후
STRICT_TIME_SPLIT 검증이 필요합니다.

`gui_app.py`는 화면과 Qt thread를, `training_engine.py`는 callback signal과
공통 학습 orchestration을 담당합니다. 핵심 전처리·모델·평가 함수는
`train_lstm_ae.py`에서 재사용합니다.

실행 모드
---------
스크립트 상단의 `MODE`로 선택합니다.

- `KAMP_REPRODUCTION`: KAMP 가이드북 방식. 기본값
- `BASELINE`: 기존 간소화 baseline

`KAMP_REPRODUCTION`은 `abs()` 전처리, 정상 앞 15,000행 Train, sequence length
20, prediction horizon 100, 마지막 timestep MSE, Precision-Recall 교차 threshold를
사용합니다. `Equipment_state`는 모델 입력에 포함하지 않습니다.

목적
----
KAMP 소성가공 예지보전 AI 데이터셋으로 프레스 설비 유압펌프의 이상 상태를
탐지합니다. 정상 데이터만 학습한 LSTM-AutoEncoder의 reconstruction error를
사용하며, 정상 Validation error의 99 percentile을 baseline threshold로 씁니다.

입력 데이터
-----------
train_lstm_ae.py와 같은 폴더에 다음 파일을 둡니다.

- press_data_normal.csv: Equipment_state가 모두 0인 정상 데이터
- outlier_data.csv: Equipment_state가 모두 1인 이상 데이터

필수 컬럼은 AI0_Vibration, AI1_Vibration, AI2_Current, Equipment_state입니다.
TimeStamp가 있으면 파싱 가능 여부와 시간 순서를 검사합니다. Equipment_state는
검증과 평가에만 쓰며 모델 입력에는 포함하지 않습니다.

Windows PowerShell 실행
-----------------------
이 프로젝트에서는 64-bit Python 3.11~3.13을 사용할 수 있습니다.
CUDA나 ROCm은 필요하지 않으며 GPU가 없으면 CPU로 실행됩니다.

    cd C:\Users\jugon\Desktop\kamp_pm
    py -3.12 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    python train_lstm_ae.py

`py -3.12`가 없다면 설치된 호환 Python으로 `python -m venv .venv`를 사용합니다.

Quick Test와 전체 학습
----------------------
기본값은 가이드북 재현을 위한 `QUICK_TEST = False`입니다. 이번 검증에서는
실행 직전에 이를 `True`로 바꾸어 KAMP Train 전체 sequence 구조를 유지한 채
3 epoch만 실행했습니다. 전체 KAMP 가이드북 재현은 `QUICK_TEST = False`로
실행하며 최대 epoch는 800이고
EarlyStopping과 ReduceLROnPlateau가 적용됩니다.

처리 흐름
---------
KAMP_REPRODUCTION 흐름:
원본 -> abs -> 정상 Train 15,000행 -> 정상 Train에만 scaler fit ->
20 timestep window + 100 horizon label -> Validation Normal 880 / Anomaly 300 분리 ->
LSTM AutoEncoder -> 마지막 timestep reconstruction error -> PR 교차 threshold -> 평가

BASELINE 흐름은 기존처럼 시간순 간소화 분할과 20x3 전체 reconstruction MSE,
정상 Validation 99 percentile threshold를 사용합니다.

결과 파일
---------
KAMP_REPRODUCTION 실행 시 outputs/model/kamp_lstm_autoencoder.keras,
training_history.csv, training_loss.png, pr_threshold_curve.png,
reconstruction_error_test.png, confusion_matrix.png,
anomaly_detection_results.csv, metrics.json, run_summary.txt가 생성됩니다.

검증 상태
---------
- CODE_VERIFIED: 구문과 코드/데이터 구조를 확인한 상태
- QUICK_TEST_VERIFIED: Quick Test를 실제로 끝까지 실행한 상태
- FULL_TRAINING_VERIFIED: 전체 학습을 실제로 끝까지 실행한 상태

실행하지 않은 단계의 결과는 검증 완료로 간주하지 않습니다.
