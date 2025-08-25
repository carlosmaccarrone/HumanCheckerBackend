import onnxruntime as ort
import numpy as np

SHAPES = ["Circle", "Square", "Triangle"]

class ShapeClassifier:
    def __init__(self, model_file: str):
        self.session = ort.InferenceSession(model_file)
        self.input_name = self.session.get_inputs()[0].name
        # print("✅ ONNX model loaded successfully.")

    def predict(self, img_array: np.ndarray):
        pred = self.session.run(None, {self.input_name: img_array})[0]
        pred_idx = int(np.argmax(pred))
        confidence = float(pred[0][pred_idx])
        return SHAPES[pred_idx], confidence