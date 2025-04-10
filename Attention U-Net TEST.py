import os,sys
# Determine the base directory
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
elif (__file__ is not None) and (os.path.dirname(__file__) != ''):
    BASE_DIR = os.path.dirname(__file__)
else:
    BASE_DIR = os.getcwd()

print("Base Dir:", BASE_DIR)
os.chdir(BASE_DIR)


import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageOps
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Configuration Constants
IMAGE_HEIGHT = 512  # 128,256,512,1024
IMAGE_WIDTH = 512   # 128,258,512,1024
NUM_CLASSES = 4

# Path to the saved model
MODEL_PATH = 'models/512_X_512_True_att_unet_model_val_loss_0.4386.pth'

# Mapping class indices to colors
color_to_class = {
    (0, 0, 0): 0,      # Background (Black)
    (0, 0, 255): 1,    # Stroma (Blue)
    (0, 255, 0): 2,    # Benign (Green)
    (255, 255, 0): 3   # Tumor (Yellow)
}

# Mapping class index to color for easier reverse mapping
class_to_color = {v: k for k, v in color_to_class.items()}

# Define Attention Block
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# Define Attention U-Net Model
class AttUNet(nn.Module):
    def __init__(self, num_classes):
        super(AttUNet, self).__init__()

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(3, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = CBR(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.dec4 = CBR(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.dec3 = CBR(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.dec2 = CBR(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.dec1 = CBR(128, 64)

        self.conv_last = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        enc4 = self.att4(g=dec4, x=enc4)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        enc3 = self.att3(g=dec3, x=enc3)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        enc2 = self.att2(g=dec2, x=enc2)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        enc1 = self.att1(g=dec1, x=enc1)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.dec1(dec1)

        return self.conv_last(dec1)

# Function to select an image using tkinter
def select_image():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select an Image",
                                           filetypes=[("Image files", "*.png *.jpg *.jpeg")])
    root.destroy()
    return file_path

# Function to visualize the image and predicted mask
def visualize_image_and_mask(model, image_path, device):
    model.eval()
    
    # Load and preprocess the image
    original_image = Image.open(image_path).convert("RGB")
    original_size = original_image.size
    transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor()
    ])
    image_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)
        output = F.softmax(output, dim=1)
    
    # Get the predicted mask
    _, predicted_mask = torch.max(output, 1)
    predicted_mask = predicted_mask.squeeze(0).cpu().numpy()
    
    # Map class indices back to colors
    height, width = predicted_mask.shape
    predicted_mask_image = np.zeros((height, width, 3), dtype=np.uint8)
    for class_idx, (color_rgb) in class_to_color.items():
        predicted_mask_image[predicted_mask == class_idx] = list(color_rgb)
    
    # Resize predicted mask image to original image size
    predicted_mask_image_pil = Image.fromarray(predicted_mask_image)
    predicted_mask_image_pil = predicted_mask_image_pil.resize(original_size, resample=Image.Resampling.NEAREST)
    predicted_mask_image_resized = np.array(predicted_mask_image_pil)
    
    # Apply mask on the original image
    applied_mask_image = np.array(original_image).copy()
    alpha = 0.5  # Transparency factor
    for class_idx, color_rgb in class_to_color.items():
        mask = np.all(predicted_mask_image_resized == np.array(list(color_rgb)), axis=-1)
        applied_mask_image[mask] = (alpha * np.array(list(color_rgb)) + (1 - alpha) * applied_mask_image[mask]).astype(np.uint8)

    # Display the original image, predicted mask, and applied mask
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(original_image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    
    ax[1].imshow(predicted_mask_image_resized)
    ax[1].set_title("Predicted Mask")
    ax[1].axis("off")
    
    ax[2].imshow(applied_mask_image)
    ax[2].set_title("Applied Mask on Image")
    ax[2].axis("off")
    
    plt.show()

    # Print detected classes, confidence scores, and total pixels
    detected_classes = set(predicted_mask.flatten())
    confidences = output.squeeze(0).cpu().numpy()
    confidence_scores = {class_idx: confidences[class_idx, predicted_mask == class_idx].mean() for class_idx in range(NUM_CLASSES)}
    total_pixels = {class_idx: np.sum(predicted_mask == class_idx) for class_idx in range(NUM_CLASSES)}
    
    print("\nDetected Classes, Confidence Scores, and Total Pixels:")
    for class_idx in detected_classes:
        class_name = next((name for name, idx in color_to_class.items() if idx == class_idx), "Unknown")
        print(f"Class: {class_name}, Confidence: {confidence_scores[class_idx]:.2f}, Pixels Detected: {total_pixels[class_idx]}")

# Load the model (ensure you change 'model.pth' to the path of your saved model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AttUNet(num_classes=NUM_CLASSES).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# Select an image using tkinter file dialog
image_path = select_image()

# Ensure that an image was selected
if image_path:
    visualize_image_and_mask(model, image_path, device)
else:
    print("No image selected.")