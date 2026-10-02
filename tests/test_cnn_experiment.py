import unittest
import numpy as np

class CnnExperimentTests(unittest.TestCase):
    def test_default_cnn_baseline_architecture_is_unchanged(self):
        from tensorflow.keras.layers import Conv1D, LSTM
        from model_builders import build_cnn_lstm_autoencoder

        model = build_cnn_lstm_autoencoder()
        self.assertEqual([(layer.filters, layer.kernel_size) for layer in model.layers if isinstance(layer, Conv1D)],
                         [(32, (3,)), (32, (3,))])
        self.assertEqual([layer.units for layer in model.layers if isinstance(layer, LSTM)], [64, 32, 32, 64])
        self.assertEqual(model.output_shape, (None, 20, 3))

    def test_cnn_and_denoising_architecture_settings_reach_both_sides(self):
        from tensorflow.keras.layers import Conv1D, LSTM
        from model_registry import MODEL_REGISTRY
        from training_engine import build_training_model
        from training_config import default_training_config

        counts = {}
        for model_id in ("CNN_LSTM_AUTOENCODER", "DENOISING_CNN_LSTM_AUTOENCODER"):
            for filters, bottleneck in ((32, 32), (16, 32), (32, 16), (16, 16), (64, 64)):
                with self.subTest(model_id=model_id, filters=filters, bottleneck=bottleneck):
                    config = default_training_config()
                    config.update(cnn_filters=filters, bottleneck_units=bottleneck, cnn_kernel_size=5)
                    model = build_training_model(MODEL_REGISTRY[model_id], config)
                    self.assertEqual([layer.filters for layer in model.layers if isinstance(layer, Conv1D)], [filters, filters])
                    self.assertEqual([layer.kernel_size for layer in model.layers if isinstance(layer, Conv1D)], [(5,), (5,)])
                    self.assertEqual([layer.units for layer in model.layers if isinstance(layer, LSTM)], [64, bottleneck, bottleneck, 64])
                    self.assertEqual(model.output_shape, (None, 20, 3))
                    counts[(model_id, filters, bottleneck)] = model.count_params()
            self.assertNotEqual(counts[(model_id, 32, 32)], counts[(model_id, 16, 32)])
            self.assertNotEqual(counts[(model_id, 32, 32)], counts[(model_id, 32, 16)])

    def test_cnn_and_denoising_builders_use_selected_kernel_twice(self):
        from tensorflow.keras.layers import Conv1D
        from model_registry import MODEL_REGISTRY
        from training_engine import build_training_model
        for model_id in ("CNN_LSTM_AUTOENCODER", "DENOISING_CNN_LSTM_AUTOENCODER"):
            for kernel in (3, 5, 7):
                with self.subTest(model_id=model_id, kernel=kernel):
                    model = build_training_model(MODEL_REGISTRY[model_id], {"cnn_kernel_size": kernel})
                    self.assertEqual([layer.kernel_size for layer in model.layers if isinstance(layer, Conv1D)], [(kernel,), (kernel,)])

    def test_non_cnn_builder_keeps_zero_argument_call(self):
        from types import SimpleNamespace
        from training_engine import build_training_model
        calls = []
        spec = SimpleNamespace(id="KAMP_LSTM_AE", builder=lambda: calls.append("called"))
        build_training_model(spec, {"cnn_kernel_size": 5})
        self.assertEqual(calls, ["called"])

    def test_active_training_registry_contains_only_two_models(self):
        from model_registry import MODEL_REGISTRY
        self.assertEqual([m.id for m in MODEL_REGISTRY.values() if m.status == "ACTIVE"], ["CNN_LSTM_AUTOENCODER", "DENOISING_CNN_LSTM_AUTOENCODER"])

    def test_denoising_noise_does_not_mutate_clean_input(self):
        from model_data import add_training_noise
        clean = np.full((2, 20, 3), .5, dtype=np.float32)
        noisy = add_training_noise(clean, std=.01, seed=42)
        self.assertTrue(np.array_equal(clean, np.full((2,20,3), .5, dtype=np.float32)))
        self.assertFalse(np.array_equal(clean, noisy))
        self.assertGreaterEqual(noisy.min(), 0); self.assertLessEqual(noisy.max(), 1)

    def test_denoising_builder_has_cnn_output_shape(self):
        from model_registry import MODEL_REGISTRY
        model = MODEL_REGISTRY["DENOISING_CNN_LSTM_AUTOENCODER"].builder()
        self.assertEqual(model.output_shape, (None, 20, 3))

    def test_denoising_fit_is_noisy_to_clean_and_validation_stays_clean(self):
        from model_data import prepare_fit_data
        train = np.full((2, 20, 3), .5, dtype=np.float32); target = train.copy()
        validation = np.full((1, 20, 3), .25, dtype=np.float32); validation_target = validation.copy()
        original = train.copy()
        fit_x, fit_y, valid_x, valid_y = prepare_fit_data(train, target, validation, validation_target,
                                                          denoising=True, std=.01, seed=42, batch_size=2)
        noisy, clean_target = next(iter(fit_x))
        self.assertTrue(np.array_equal(train, original))
        self.assertIsNone(fit_y)
        self.assertFalse(np.array_equal(noisy.numpy(), train))
        self.assertTrue(np.array_equal(clean_target.numpy(), target))
        self.assertTrue(np.array_equal(valid_x, validation))
        self.assertTrue(np.array_equal(valid_y, validation_target))

    def test_dynamic_noise_changes_between_passes_and_repeats_with_seed(self):
        from model_data import prepare_fit_data
        clean = np.full((2, 20, 3), .5, dtype=np.float32)
        def make_dataset():
            return prepare_fit_data(clean, clean, clean, clean, denoising=True, std=.01, seed=123, batch_size=2)[0]
        dataset = make_dataset()
        first = next(iter(dataset))[0].numpy()
        second = next(iter(dataset))[0].numpy()
        self.assertFalse(np.array_equal(first, second))
        self.assertTrue(np.array_equal(first, next(iter(make_dataset()))[0].numpy()))

    def test_zero_std_and_clip_and_gui_noise_settings(self):
        from model_data import prepare_fit_data
        clean = np.full((2, 20, 3), .5, dtype=np.float32)
        zero = prepare_fit_data(clean, clean, clean, clean, denoising=True, mean=.1, std=0, seed=7, batch_size=2)[0]
        self.assertTrue(np.array_equal(next(iter(zero))[0].numpy(), clean))
        clipped = prepare_fit_data(clean, clean, clean, clean, denoising=True, mean=.8, std=.1, clip=True, seed=7, batch_size=2)[0]
        self.assertLessEqual(float(next(iter(clipped))[0].numpy().max()), 1.0)
        unclipped = prepare_fit_data(clean, clean, clean, clean, denoising=True, mean=.8, std=.1, clip=False, seed=7, batch_size=2)[0]
        self.assertGreater(float(next(iter(unclipped))[0].numpy().min()), 1.0)

    def test_baseline_stays_array_based(self):
        from model_data import prepare_fit_data
        clean = np.full((2, 20, 3), .5, dtype=np.float32)
        fit_x, fit_y, valid_x, valid_y = prepare_fit_data(clean, clean, clean, clean)
        self.assertIsInstance(fit_x, np.ndarray)
        self.assertTrue(np.array_equal(fit_x, fit_y))
        self.assertTrue(np.array_equal(valid_x, valid_y))

    def test_evaluation_task_bundle_keeps_clean_features(self):
        import pandas as pd
        import train_lstm_ae as core
        from model_data import create_task_bundle
        from model_registry import MODEL_REGISTRY
        values = np.linspace(.1, .9, 150, dtype=np.float32)
        data = pd.DataFrame({name: values for name in core.FEATURES})
        data[core.LABEL_COLUMN] = 0
        bundle = create_task_bundle(data, MODEL_REGISTRY["DENOISING_CNN_LSTM_AUTOENCODER"])
        self.assertTrue(np.array_equal(bundle.inputs[0], data[core.FEATURES].to_numpy(dtype=np.float32)[:20]))
        self.assertTrue(np.array_equal(bundle.inputs, bundle.targets))

if __name__ == "__main__": unittest.main()
