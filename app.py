from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from io import BytesIO
import os

app = Flask(__name__)
CORS(app)

# ---------------- DEVICE ----------------
device = torch.device("cpu")

# SINGLE MODEL CLASSES
classes = ["crazing", "shrinkage", "structural", "thermal"]

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ---------------- KNOWLEDGE BASE ----------------
causes = {
    "structural": "Possible load stress or foundation issue",
    "thermal": "Temperature variation caused expansion/contraction",
    "shrinkage": "Improper curing or drying",
    "crazing": "Surface-level fine cracks due to finishing"
}

fixes = {
    "structural": "Epoxy injection or structural reinforcement",
    "thermal": "Provide expansion joints or flexible sealant",
    "shrinkage": "Seal cracks and improve curing practices",
    "crazing": "Surface sealing or coating"
}

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return "Crack Detection API is running"


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        # Load image
        image = Image.open(BytesIO(file.read())).convert("RGB")
        image = transform(image).unsqueeze(0)

        # -------- LOAD SINGLE MODEL --------
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, 4)
        model.load_state_dict(torch.load("crack_classifier.pth", map_location=device))
        model = model.to(device)
        model.eval()

        # -------- PREDICTION --------
        with torch.no_grad():
            out = model(image)
            prob = torch.softmax(out, dim=1)
            _, pred = torch.max(out, 1)
            label = classes[pred.item()]
            confidence = prob[0][pred.item()].item()

        # -------- NO CRACK HEURISTIC --------
        if confidence < 0.5:
            return jsonify({
                "status": "no_crack",
                "message": "No clear crack detected",
                "confidence": round(confidence * 100, 2)
            })

        # -------- CRACK OUTPUT --------
        return jsonify({
            "status": "crack",
            "type": label,
            "confidence": round(confidence * 100, 2),
            "cause": causes[label],
            "fix": fixes[label]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
