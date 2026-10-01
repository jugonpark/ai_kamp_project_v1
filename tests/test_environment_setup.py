import importlib
from importlib.metadata import version
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class EnvironmentSetupTests(unittest.TestCase):
    def test_pinned_runtime_requirements_import_in_current_venv(self):
        package_modules = {
            "tensorflow": "tensorflow", "numpy": "numpy", "pandas": "pandas",
            "scipy": "scipy", "scikit-learn": "sklearn", "matplotlib": "matplotlib",
            "PySide6": "PySide6",
        }
        pins = {}
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                name, pinned_version = line.split("==")
                pins[name] = pinned_version
        self.assertEqual(set(pins), set(package_modules))
        for package, module in package_modules.items():
            importlib.import_module(module)
            self.assertEqual(version(package), pins[package])

    def test_environment_verification_accepts_cpu_without_gpu(self):
        from verify_environment import verify
        ok, lines = verify()
        self.assertTrue(ok, "\n".join(lines))
        self.assertTrue(any(line.startswith("TensorFlow CPU devices:") for line in lines))
        self.assertTrue(any(line.startswith("Project import gui_app: OK") for line in lines))
        self.assertTrue(any(line.startswith("Project import model_data: OK") for line in lines))

    def test_launchers_use_quoted_script_relative_paths(self):
        setup = (ROOT / "setup_environment.bat").read_text(encoding="utf-8")
        normal = (ROOT / "run_gui.bat").read_text(encoding="utf-8")
        debug = (ROOT / "run_gui_debug.bat").read_text(encoding="utf-8")
        for script in (setup, normal, debug):
            self.assertIn('cd /d "%~dp0"', script)
            self.assertIn("chcp 65001", script)
            self.assertIn('set "PYTHONIOENCODING=utf-8"', script)
        self.assertIn('"%~dp0.venv\\Scripts\\pythonw.exe" "%~dp0gui_app.py"', normal)
        self.assertIn('"%~dp0.venv\\Scripts\\python.exe" "%~dp0gui_app.py"', debug)
        self.assertIn('"%VENV_PY%" -m pip install --only-binary=:all: -r "%~dp0requirements.txt"', setup)
        self.assertIn('"%VENV_PY%" "%~dp0verify_environment.py"', setup)
        self.assertIn("py -3.12", setup)
        self.assertIn("python -c", setup)
        self.assertIn("py -c", setup)


if __name__ == "__main__":
    unittest.main()
