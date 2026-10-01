import unittest
import numpy as np

class CnnExperimentTests(unittest.TestCase):
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
                                                          denoising=True, std=.01, seed=42)
        self.assertTrue(np.array_equal(train, original))
        self.assertFalse(np.array_equal(fit_x, train))
        self.assertTrue(np.array_equal(fit_y, target))
        self.assertTrue(np.array_equal(valid_x, validation))
        self.assertTrue(np.array_equal(valid_y, validation_target))

if __name__ == "__main__": unittest.main()
