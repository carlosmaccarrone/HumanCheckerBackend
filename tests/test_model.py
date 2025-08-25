from backend.model import ShapeClassifier, SHAPES
from unittest.mock import patch, MagicMock
import numpy as np
import unittest

class TestShapeClassifier(unittest.TestCase):
    @patch("backend.model.ort.InferenceSession")
    def test_model_init_and_predict(self, mock_inference_session):
        # ONNX Session Mock
        mock_session = MagicMock()
        # fake input name
        mock_session.get_inputs.return_value = [MagicMock(name="input", spec=["name"])]
        mock_session.get_inputs.return_value[0].name = "fake_input"
        # fake output: always predicts "Square" with 0.85 confidence
        mock_session.run.return_value = [np.array([[0.1, 0.85, 0.05]])]

        mock_inference_session.return_value = mock_session

        # create classifier with mock
        clf = ShapeClassifier("fake_model.onnx")

        # input fake
        fake_input = np.zeros((1, 28, 28, 1), dtype=np.float32)
        pred_shape, confidence = clf.predict(fake_input)

        self.assertIn(pred_shape, SHAPES)
        self.assertEqual(pred_shape, "Square")
        self.assertAlmostEqual(confidence, 0.85, places=2)

        # check that the mocked input name has been used
        mock_session.run.assert_called_once_with(None, {"fake_input": fake_input})

if __name__ == "__main__":
    unittest.main()