from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app import app
import unittest

client = TestClient(app)

class TestApp(unittest.TestCase):
    def test_head_request(self):
        """endpoint responds to the HEAD correctly."""
        response = client.head("/predict")
        self.assertEqual(response.status_code, 200)
        # HEAD normally has no body
        self.assertEqual(response.text, "")

    def test_post_without_file(self):
        """If I don't send a file, it should return ping without an image."""
        response = client.post("/predict", files={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "Ping received, no image")

    @patch("app.classifier")  # mock the model
    @patch("app.preprocess_image")  # mock preprocess to avoid loading anything real
    def test_post_with_file(self, mock_preprocess, mock_classifier):
        """with file uploaded, it should invoke the model and return prediction."""
        # fake preprocess: return a dummy array
        mock_preprocess.return_value = [[1, 2, 3]]

        # fake model: return form + confidence
        mock_classifier.predict.return_value = ("Circle", 0.99)

        file_content = b"fake image content"
        response = client.post(
            "/predict",
            files={"file": ("test.png", file_content, "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn("predicted_shape", json_data)
        self.assertIn("confidence", json_data)
        self.assertEqual(json_data["predicted_shape"], "Circle")
        self.assertEqual(json_data["confidence"], 0.99)

if __name__ == "__main__":
    unittest.main()