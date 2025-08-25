[![Python CI](https://github.com/carlosmaccarrone/HumanCheckerBackend/actions/workflows/ci.yml/badge.svg)](https://github.com/carlosmaccarrone/HumanCheckerBackend/actions/workflows/ci.yml)

# Human Checker Backend

This is the backend for the **Human Checker App**, a React-based application that classifies **squares, circles, and triangles** based on the user's canvas input. The backend is built with **FastAPI** and leverages **ONNX Runtime** for extremely fast model inference.

## Features

- Accepts image uploads from the frontend canvas.
- Preprocesses images to match the input format of the classifier.
- Predicts shape (`Circle`, `Square`, or `Triangle`) and returns confidence scores.
- Uses **ONNX Runtime** for high-performance inference, even though the model was originally trained in TensorFlow. This approach ensures **low latency** and avoids heavy TensorFlow dependencies.

## Backend Stack

- **FastAPI** – lightweight API framework
- **uvicorn** – ASGI server
- **ONNX Runtime** – high-speed model inference
- **NumPy** & **Pillow** – image processing
- **Python-Multipart** – handling file uploads
- **httpx** – testing and HTTP client utilities

## Getting Started

### Requirements

- Python 3.11+
- `shapes_model.onnx` in the backend folder
- Recommended: virtual environment

### Installation

```bash
$ git clone https://github.com/carlosmaccarrone/HumanCheckerBackend.git
$ cd HumanCheckerBackend
$ pip install virtualenv
$ virtualenv venv
$ .\venv\Scripts\activate
$ pip install fastapi uvicorn[standard] onnxruntime numpy pillow python-multipart httpx
```

### Running the Server

```
uvicorn app:app --reload --port 8001
```
The backend will start at http://127.0.0.1:8001.

###Endpoints

`POST /predict`  
Description: Classifies a shape from the uploaded canvas image.  
Request: multipart/form-data with a file field named file  

Response:
```
{
  "predicted_shape": "Circle",
  "confidence": 0.95
}
```

`HEAD /predict` or `POST /predict` without file  
Returns a simple ping message:  
```
{
  "message": "Ping received, no image"
}
```

### Testing

The backend uses unittest for testing, including tests for:  
- Image preprocessing  
- Model prediction (mocked)  
- API endpoints  

Run tests:
```
python -m unittest discover -s tests -v
```

### Notes

- The shapes_model.onnx file is the optimized inference model exported from TensorFlow. Using ONNX Runtime significantly improves speed for real-time predictions compared to running the original TensorFlow model.  

- The backend is designed to be lightweight and fast, suitable for integration with the React frontend.  









