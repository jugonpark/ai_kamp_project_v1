"""Check the Windows CPU-first runtime without starting the GUI or training."""
from datetime import datetime
import importlib
import os
from pathlib import Path
import platform
import sys


PACKAGES = (
    ("TensorFlow", "tensorflow"),
    ("NumPy", "numpy"),
    ("Pandas", "pandas"),
    ("SciPy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("Matplotlib", "matplotlib"),
    ("PySide6", "PySide6"),
)
PROJECT_MODULES = (
    "gui_app", "training_engine", "training_config", "model_builders",
    "model_data", "anomaly_scoring", "threshold_methods",
)


def verify() -> tuple[bool, list[str]]:
    lines = [
        f"Python version: {platform.python_version()}",
        f"Python executable: {sys.executable}",
        f"OS: {platform.platform()}",
        f"Architecture: {platform.architecture()[0]}",
        f"CPU: {platform.processor() or os.environ.get('PROCESSOR_IDENTIFIER', 'unknown')}",
    ]
    valid = sys.version_info[:2] in ((3, 12), (3, 13)) and sys.maxsize > 2**32
    if not valid:
        lines.append("ERROR: Python 3.12 또는 3.13 64-bit가 필요합니다.")

    loaded = {}
    for label, module_name in PACKAGES:
        try:
            module = importlib.import_module(module_name)
            loaded[module_name] = module
            lines.append(f"{label}: {module.__version__}")
        except Exception as exc:
            valid = False
            lines.append(f"{label}: IMPORT FAILED ({type(exc).__name__}: {exc})")

    tf = loaded.get("tensorflow")
    if tf is not None:
        try:
            cpu = tf.config.list_physical_devices("CPU")
            gpu = tf.config.list_physical_devices("GPU")
            lines.append(f"TensorFlow CPU devices: {[device.name for device in cpu]}")
            lines.append(f"TensorFlow GPU devices: {[device.name for device in gpu]}")
            if not gpu:
                lines.append("GPU 미사용 / CPU 모드로 실행 가능")
        except Exception as exc:
            valid = False
            lines.append(f"TensorFlow device check failed: {type(exc).__name__}: {exc}")

    for name in PROJECT_MODULES:
        try:
            importlib.import_module(name)
            lines.append(f"Project import {name}: OK")
        except Exception as exc:
            valid = False
            lines.append(f"Project import {name}: FAILED ({type(exc).__name__}: {exc})")
    lines.append("환경 검증: 성공" if valid else "환경 검증: 실패")
    return valid, lines


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    valid, lines = verify()
    for line in lines:
        print(line, flush=True)
    report_dir = Path(__file__).resolve().parent / "outputs" / "environment"
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        report = report_dir / f"environment_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.txt"
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"환경 보고서: {report}")
    except OSError as exc:
        print(f"환경 보고서를 저장하지 못했습니다: {exc}")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
