import unittest

import numpy as np

from score_postprocessing import (
    EWMAProcessor,
    apply_temporal_processing,
    detect_timestamp_segments,
    fp_fn_pareto_flags,
)


class TemporalScoreTests(unittest.TestCase):
    def test_none_is_exact_identity_even_with_boundaries(self):
        scores = np.array([1, 2, 3], dtype=np.int64)
        actual = apply_temporal_processing(scores, segment_ids=[0, 0, 1],
                                           contains_timestamp_gap=[False, True, False])
        np.testing.assert_array_equal(actual, scores)
        self.assertEqual(actual.dtype, scores.dtype)

    def test_known_vector_and_causality(self):
        np.testing.assert_allclose(
            apply_temporal_processing([1, 2, 3], method="EWMA", alpha=.5),
            [1, 1.5, 2.25],
        )
        first = apply_temporal_processing([1, 2, 3], method="EWMA", alpha=.5)
        changed_future = apply_temporal_processing([1, 2, 999], method="EWMA", alpha=.5)
        np.testing.assert_array_equal(first[:2], changed_future[:2])

    def test_alpha_validation(self):
        for value in (0, -1, 1.01, np.nan, np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                EWMAProcessor(value)
        for value in (.02, .4, 1):
            EWMAProcessor(value)

    def test_segment_and_gap_reset(self):
        scores = [1, 1, 1, 10, 10]
        by_id = apply_temporal_processing(scores, "EWMA", .2, segment_ids=[0, 0, 0, 1, 1])
        by_gap = apply_temporal_processing(scores, "EWMA", .2,
                                           contains_timestamp_gap=[False, False, False, True, False])
        np.testing.assert_array_equal(by_id, [1, 1, 1, 10, 10])
        np.testing.assert_array_equal(by_gap, by_id)

    def test_independent_calls_and_stateful_reset(self):
        processor = EWMAProcessor(.5)
        self.assertEqual(processor.update(1), 1)
        self.assertEqual(processor.update(3), 2)
        processor.reset()
        self.assertEqual(processor.update(10), 10)
        validation = apply_temporal_processing([1, 3], "EWMA", .5)
        test = apply_temporal_processing([10, 10], "EWMA", .5)
        self.assertEqual(validation[-1], 2)
        self.assertEqual(test[0], 10)

    def test_timestamp_gap_and_nonmonotonic_boundary(self):
        ids, threshold = detect_timestamp_segments([0, .1, .2, 1, 1.1, 1.0, 1.1])
        np.testing.assert_array_equal(ids, [0, 0, 0, 1, 1, 2, 2])
        self.assertAlmostEqual(threshold, .15)

    def test_datetime64_and_pandas_series(self):
        import pandas as pd
        timestamps = pd.Series(pd.to_datetime([
            "2026-10-02T00:00:00.000", "2026-10-02T00:00:00.100",
            "2026-10-02T00:00:00.200", "2026-10-02T00:00:01.000",
        ]))
        ids, threshold = detect_timestamp_segments(timestamps)
        np.testing.assert_array_equal(ids, [0, 0, 0, 1])
        self.assertAlmostEqual(threshold, .15, places=5)

    def test_invalid_shapes(self):
        with self.assertRaises(ValueError):
            apply_temporal_processing([[1, 2]], "EWMA")
        with self.assertRaises(ValueError):
            apply_temporal_processing([1, 2], "EWMA", segment_ids=[0])
        with self.assertRaises(ValueError):
            apply_temporal_processing([1, 2], "EWMA", contains_timestamp_gap=[False])

    def test_fp_fn_pareto_keeps_tradeoffs_and_ties(self):
        flags = fp_fn_pareto_flags([5, 4, 6, 4, 7], [5, 7, 4, 7, 7])
        np.testing.assert_array_equal(flags, [True, True, True, True, False])


if __name__ == "__main__":
    unittest.main()
