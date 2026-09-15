import os
import io
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify
from pathlib import Path

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "checkpoints" / "best_model.pth"
CLASSES = ["mild", "moderate", "severe", "healthy"]
NUM_CLASSES = 4
IMG_SIZE = 224
DEVICE = torch.device("cpu")

SPRAY_INFO = {
    "healthy": {
        "status": "HEALTHY",
        "color": "#22c55e",
        "icon": "healthy",
        "action": "No Spray Required",
        "description": "Plant is healthy. No pesticide treatment needed.",
        "duration": "0 sec",
        "flow_rate": "0 LPM",
        "recommendation": "Continue regular monitoring.",
    },
    "mild": {
        "status": "MILD INFECTION",
        "color": "#f59e0b",
        "icon": "mild",
        "action": "Low Spray Recommended",
        "description": "Early signs of disease detected. Short-duration targeted spray recommended.",
        "duration": "2 sec",
        "flow_rate": "0.5 LPM",
        "recommendation": "Apply low-dose fungicide. Re-check in 3-5 days.",
    },
    "moderate": {
        "status": "MODERATE INFECTION",
        "color": "#f97316",
        "icon": "moderate",
        "action": "Medium Spray Required",
        "description": "Significant disease spread detected. Medium-duration spray needed.",
        "duration": "5 sec",
        "flow_rate": "1.0 LPM",
        "recommendation": "Apply standard fungicide dosage. Monitor daily.",
    },
    "severe": {
        "status": "SEVERE INFECTION",
        "color": "#ef4444",
        "icon": "severe",
        "action": "High Spray Urgent",
        "description": "Critical disease level. Immediate high-duration spray required.",
        "duration": "10 sec",
        "flow_rate": "2.0 LPM",
        "recommendation": "Apply maximum dosage. Isolate affected area. Re-treat in 2 days.",
    },
}

model: Optional[nn.Module] = None
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model() -> nn.Module:
    global model, CLASSES, NUM_CLASSES
    if model is not None:
        return model

    # Build the SAME architecture used by train.py so the checkpoint loads.
    from torchvision.models import mobilenet_v2

    net: nn.Module = mobilenet_v2(weights=None)
    net.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(net.last_channel, NUM_CLASSES),
    )
    model = net

    if MODEL_PATH.exists():
        checkpoint = torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"Model loaded from {MODEL_PATH}")

        names = checkpoint.get("class_names")
        if names:
            CLASSES = list(names)
            NUM_CLASSES = len(names)
            print(f"Model classes: {CLASSES}")
    else:
        print(f"WARNING: No model found at {MODEL_PATH}")

    model.to(DEVICE)
    model.eval()
    return model


def predict(image_bytes):
    net = load_model()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img)
    assert isinstance(tensor, torch.Tensor)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs: torch.Tensor = net(tensor)
        probs: torch.Tensor = F.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)

    predicted_class = CLASSES[int(predicted.item())]
    confidence_score = float(confidence.item())
    all_probs = {CLASSES[i]: round(float(probs[0][i].item()) * 100, 1) for i in range(NUM_CLASSES)}

    result = {
        "class": predicted_class,
        "confidence": round(confidence_score * 100, 1),
        "probabilities": all_probs,
        "spray": SPRAY_INFO[predicted_class],
    }
    return result


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    image_bytes = file.read()
    if len(image_bytes) == 0:
        return jsonify({"error": "Empty file"}), 400

    try:
        result = predict(image_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict_base64", methods=["POST"])
def predict_base64():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image data"}), 400

    try:
        header, encoded = data["image"].split(",", 1)
        image_bytes = base64.b64decode(encoded)
        result = predict(image_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_model()
    print("\nStarting Spinach Health Detector...")
    print("Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
