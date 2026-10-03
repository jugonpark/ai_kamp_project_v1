import unittest
from types import SimpleNamespace
from unittest.mock import patch

from preprocessing_config import default_preprocessing_config
from preprocessing_worker import PreprocessingWorker


class PreprocessingWorkerTests(unittest.TestCase):
    def test_analyze_only_calls_quality_backend(self):
        config = default_preprocessing_config()
        worker = PreprocessingWorker("analyze", "normal.csv", "anomaly.csv", config)
        reports, failures, finished = [], [], []
        worker.analysis_ready.connect(reports.append)
        worker.failed.connect(failures.append)
        worker.finished.connect(lambda: finished.append(True))
        with patch("preprocessing_worker.inspect_csv") as inspect, patch(
                "preprocessing_worker.run_stage2_preprocessing") as preprocess, patch(
                "preprocessing_worker.window_count_summary", return_value={
                    "windows": 10, "eligible_segments": 2}) as counts:
            inspect.side_effect = [SimpleNamespace(report={"segment_count": 599}, frame=object()),
                                   SimpleNamespace(report={"segment_count": 21}, frame=object())]
            worker.run()
        self.assertEqual(reports, [{"normal": {"segment_count": 599, "selected_sequence_windows": {
            "windows": 10, "eligible_segments": 2}}, "anomaly": {"segment_count": 21,
            "selected_sequence_windows": {"windows": 10, "eligible_segments": 2}}}])
        self.assertEqual(counts.call_count, 2)
        self.assertEqual([call.kwargs["expected_label"] for call in inspect.call_args_list], [0, 1])
        preprocess.assert_not_called()
        self.assertEqual(failures, [])
        self.assertEqual(finished, [True])

    def test_preprocess_forwards_paths_and_config_and_reports_errors(self):
        config = {**default_preprocessing_config(), "sequence_length": 15,
                  "signal_transform": "RAW_SIGNED", "scaler": "STANDARD"}
        worker = PreprocessingWorker("preprocess", "normal.csv", "anomaly.csv", config)
        results, failures = [], []
        worker.preprocessing_ready.connect(results.append)
        worker.failed.connect(failures.append)
        expected = object()
        with patch("preprocessing_worker.run_stage2_preprocessing", return_value=expected) as backend:
            worker.run()
        backend.assert_called_once_with("normal.csv", "anomaly.csv", config)
        self.assertEqual(results, [expected])
        with patch("preprocessing_worker.run_stage2_preprocessing", side_effect=FileNotFoundError("missing.csv")):
            worker.run()
        self.assertIn("FileNotFoundError: missing.csv", failures[0])
