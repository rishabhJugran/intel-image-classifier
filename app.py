import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Load model definition
class CNNModel(nn.Module):
    
    def __init__(self, inputs:int, hidden_neuron:int,outputs:int):
        super(CNNModel, self).__init__()
        
        self.conv_1 = nn.Conv2d(inputs,hidden_neuron,kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv_2 = nn.Conv2d(hidden_neuron,hidden_neuron*2,kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(hidden_neuron*2,hidden_neuron*4,kernel_size=3,padding=1)
        
        self.fc1 = nn.Linear(hidden_neuron*4*18*18, 256)
        self.fc2 = nn.Linear(256, outputs)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        
        x = self.pool(self.relu(self.conv_1(x)))
        x = self.pool(self.relu(self.conv_2(x)))
        x = self.pool(self.relu(self.conv_3(x)))
        
        x = x.view(-1, 32*4*18*18)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x
    

# Load trained model
model = CNNModel(inputs=3,hidden_neuron=32,outputs=6)
model.load_state_dict(torch.load("intel_image_classifier_cnn_model.pth", map_location="cpu"))
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# Streamlit UI
st.title("Intel Image Classification 🖼️")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

classes = ["Buildings", "Forest", "Glacier", "Mountain", "Sea", "Street"]

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    st.write(f"### Prediction: {classes[predicted.item()]}")
