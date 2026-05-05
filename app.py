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

binary_classes = ["crack", "no_crack"]
multi_classes = ["crazing", "shrinkage", "structural", "thermal"]

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
        image = Image.open(BytesIO(file.read())).convert("RGB")
        image = transform(image).unsqueeze(0)

        # -------- LOAD BINARY MODEL (ONLY WHEN NEEDED) --------
        binary_model = models.resnet18()
        binary_model.fc = nn.Linear(binary_model.fc.in_features, 2)
        binary_model.load_state_dict(torch.load("crack_detector.pth", map_location=device))
        binary_model = binary_model.to(device)
        binary_model.eval()

        with torch.no_grad():
            out = binary_model(image)
            prob = torch.softmax(out, dim=1)
            _, pred = torch.max(out, 1)
            confidence = prob[0][pred.item()].item()

        if binary_classes[pred.item()] == "no_crack":
            return jsonify({
                "status": "no_crack",
                "message": "No crack detected",
                "confidence": round(confidence * 100, 2)
            })

        # -------- LOAD MULTICLASS MODEL ONLY IF CRACK --------
        multi_model = models.resnet18()
        multi_model.fc = nn.Linear(multi_model.fc.in_features, 4)
        multi_model.load_state_dict(torch.load("crack_classifier.pth", map_location=device))
        multi_model = multi_model.to(device)
        multi_model.eval()

        with torch.no_grad():
            out = multi_model(image)
            prob = torch.softmax(out, dim=1)
            _, pred = torch.max(out, 1)
            crack_type = multi_classes[pred.item()]
            confidence = prob[0][pred.item()].item()

        return jsonify({
            "status": "crack",
            "type": crack_type,
            "confidence": round(confidence * 100, 2),
            "cause": causes[crack_type],
            "fix": fixes[crack_type]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
