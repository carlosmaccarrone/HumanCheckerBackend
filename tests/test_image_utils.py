from backend.image_utils import preprocess_image
from io import BytesIO
from PIL import Image
import numpy as np
import unittest

class TestImageUtils(unittest.TestCase):
    def setUp(self):
        # create an image in memory (blank)
        img = Image.new("L", (28, 28), color=255)
        self.img_bytes = BytesIO()
        img.save(self.img_bytes, format="PNG")
        self.img_bytes.seek(0)

    def test_preprocess_image_shape_and_type(self):
        arr = preprocess_image(self.img_bytes)
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, (1, 28, 28, 1))
        self.assertEqual(arr.dtype, np.float32)

    def test_preprocess_image_values(self):
        arr = preprocess_image(self.img_bytes)
        unique_vals = np.unique(arr)
        # since it is binarized, it should only have 0.0 and/or 1.0
        for val in unique_vals:
            self.assertIn(val, [0.0, 1.0])

if __name__ == "__main__":
    unittest.main()