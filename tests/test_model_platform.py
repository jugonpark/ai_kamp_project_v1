import unittest
import numpy as np
import pandas as pd


class ModelPlatformTests(unittest.TestCase):
    def test_registry_has_four_models_and_kamp_uses_legacy_builder(self):
        import train_lstm_ae as core
        from model_registry import MODEL_REGISTRY
        self.assertEqual(set(MODEL_REGISTRY), {"KAMP_LSTM_AE", "GRU_AUTOENCODER", "CNN_LSTM_AUTOENCODER", "DENOISING_CNN_LSTM_AUTOENCODER", "LSTM_FORECAST_5"})
        self.assertIs(MODEL_REGISTRY["KAMP_LSTM_AE"].builder, core.build_model)

    def test_all_model_output_shapes(self):
        from model_registry import MODEL_REGISTRY
        expected = {"KAMP_LSTM_AE": (None,20,3), "GRU_AUTOENCODER": (None,20,3),
                    "CNN_LSTM_AUTOENCODER": (None,20,3), "DENOISING_CNN_LSTM_AUTOENCODER": (None,20,3), "LSTM_FORECAST_5": (None,5,3)}
        for model_id, shape in expected.items():
            model = MODEL_REGISTRY[model_id].builder()
            self.assertEqual(model.output_shape, shape)
            self.assertEqual(model(np.zeros((1,20,3), np.float32)).shape, (1,*shape[1:]))

    def test_forecast_alignment_matches_reconstruction_sample_count(self):
        from model_data import create_task_bundle
        from model_registry import MODEL_REGISTRY
        rows = 130
        frame = pd.DataFrame({"AI0_Vibration":np.arange(rows), "AI1_Vibration":np.arange(rows)+1000,
                              "AI2_Current":np.arange(rows)+2000, "Equipment_state":np.arange(rows)%2})
        reconstruction = create_task_bundle(frame, MODEL_REGISTRY["KAMP_LSTM_AE"])
        forecast = create_task_bundle(frame, MODEL_REGISTRY["LSTM_FORECAST_5"])
        self.assertEqual(len(reconstruction.inputs), len(forecast.inputs))
        np.testing.assert_array_equal(forecast.inputs[0,:,0], np.arange(20))
        np.testing.assert_array_equal(forecast.targets[0,:,0], np.arange(20,25))
        self.assertEqual(forecast.labels[0], frame.iloc[120].Equipment_state)


if __name__ == "__main__": unittest.main()
