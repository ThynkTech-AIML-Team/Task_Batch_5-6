
from flask import Flask, request, render_template_string
from PIL import Image
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

app = Flask(__name__)

HTML = """
<!doctype html>
<title>MNIST Image Classifier</title>
<h2>MNIST Digit Classifier (Upload an image)</h2>
<form method=post enctype=multipart/form-data>
  <input type=file name=file accept="image/*">
  <input type=submit value="Predict">
</form>

{% if pred is not none %}
  <hr>
  <h3>Prediction: {{ pred }}</h3>
  <p>Confidence: {{ conf }}</p>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def home():
    pred = None
    conf = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template_string(HTML, pred=None, conf=None)

        f = request.files["file"]
        if f.filename == "":
            return render_template_string(HTML, pred=None, conf=None)

        img_bytes = f.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            p, y = torch.max(probs, dim=1)

        pred = int(y.item())
        conf = f"{float(p.item()):.3f}"

    return render_template_string(HTML, pred=pred, conf=conf)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
