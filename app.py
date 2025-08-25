from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from backend.model import ShapeClassifier
from backend.image_utils import preprocess_image

# -----------------------------
# FastAPI Configuration
# -----------------------------
app = FastAPI(title="Shape Classifier")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://carlosmaccarrone.github.io"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Initialize model
# -----------------------------
classifier = ShapeClassifier(model_file="shapes_model.onnx")

# -----------------------------
# Endpoints
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_array = preprocess_image(file.file)
    pred_shape, confidence = classifier.predict(img_array)
    return {"predicted_shape": pred_shape, "confidence": confidence}