import unittest

from stage2.segment_detection import summarize_segment_details


class SegmentCoverageTests(unittest.TestCase):
    def test_coverage_and_detection_rate_use_distinct_denominators(self):
        rows = [{"evaluable": index < 10, "detected": index < 9,
                 "delay_from_segment_start_seconds": float(index) if index < 9 else None,
                 "delay_after_first_evaluable_seconds": float(index) if index < 9 else None}
                for index in range(11)]
        result = summarize_segment_details(rows)
        self.assertEqual(result["segment_coverage"], 10 / 11)
        self.assertEqual(result["segment_detection_rate"], 9 / 10)
        self.assertEqual(result["non_evaluable_anomaly_segments"], 1)

    def test_zero_denominators(self):
        empty = summarize_segment_details([])
        self.assertIsNone(empty["segment_coverage"])
        self.assertIsNone(empty["segment_detection_rate"])


if __name__ == "__main__": unittest.main()
