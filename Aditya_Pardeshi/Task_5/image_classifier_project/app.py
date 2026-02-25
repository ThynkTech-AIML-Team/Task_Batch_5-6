import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CIFAR10CNN
import torch.nn.functional as F

# Class names
classes = ['airplane', 'automobile', 'bird',
           'cat', 'deer', 'dog',
           'frog', 'horse', 'ship', 'truck']

# Load model
model = CIFAR10CNN()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),
                         (0.5,0.5,0.5))
])

st.title("CIFAR-10 Image Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    st.write(f"Prediction: **{classes[predicted.item()]}**")
    st.write(f"Confidence: **{confidence.item()*100:.2f}%**")