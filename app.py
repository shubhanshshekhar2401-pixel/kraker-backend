from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

device = torch.device("cpu")

# ---------------- LOAD MODELS ----------------
binary_model = models.resnet18()
binary_model.fc = nn.Linear(binary_model.fc.in_features, 2)
binary_model.load_state_dict(torch.load("crack_detector.pth", map_location=device))
binary_model.eval()

multi_model = models.resnet18()
multi_model.fc = nn.Linear(multi_model.fc.in_features, 4)
multi_model.load_state_dict(torch.load("crack_classifier.pth", map_location=device))
multi_model.eval()

binary_classes = ["crack", "no_crack"]
multi_classes = ["crazing", "shrinkage", "structural", "thermal"]

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ---------------- LOGIC ----------------
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

# ---------------- ROUTE ----------------
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    filepath = "temp.jpg"
    file.save(filepath)

    image = Image.open(filepath).convert("RGB")
    image = transform(image).unsqueeze(0)

    # Binary
    with torch.no_grad():
        out = binary_model(image)
        prob = torch.softmax(out, dim=1)
        _, pred = torch.max(out, 1)
        confidence = prob[0][pred.item()].item()

    if binary_classes[pred.item()] == "no_crack":
        return jsonify({
            "status": "no_crack",
            "message": "No crack detected"
        })

    # Multiclass
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

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
