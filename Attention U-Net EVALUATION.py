import sys, os
# Determine the base directory
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
elif (__file__ is not None) and (os.path.dirname(__file__) != ''):
    BASE_DIR = os.path.dirname(__file__)
else:
    BASE_DIR = os.getcwd()

print("Base Dir:", BASE_DIR)
os.chdir(BASE_DIR)

from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt

# Constants and Parameters
VAL_IMAGE_DIR = 'Raw datasets/Segmentation/Validation'
IMAGE_HEIGHT = 512 #128,258,512,1024
IMAGE_WIDTH = 512 #128,258,512,1024
BATCH_SIZE = 4
NUM_CLASSES = 4
MODEL_SAVE_PATH = 'models/512_X_512_True_att_unet_model_val_loss_0.4386.pth'

# Map colors to class indices
color_to_class = {
    (0, 0, 0): 0,      # Background (Black)
    (0, 0, 255): 1,    # Stroma (Blue)
    (0, 255, 0): 2,    # Benign (Green)
    (255, 255, 0): 3   # Tumor (Yellow)
}

class_to_color = {v: k for k, v in color_to_class.items()}

# Define transformations for validation images
image_transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor()
])
mask_transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=Image.NEAREST),
])

# Segmentation dataset class definition
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.png') and not f.endswith('_mask.png')]
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.image_dir, img_name.replace('.png', '_mask.png'))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        mask = self.to_class_indices(mask)

        return image, mask

    def to_class_indices(self, mask):
        mask = np.array(mask)
        class_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
        for rgb, class_idx in color_to_class.items():
            indices = np.all(mask == rgb, axis=-1)
            class_mask[indices] = class_idx
        return torch.from_numpy(class_mask)

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

# Calculate IoU for a single class
def calculate_iou(pred_mask, true_mask, class_idx):
    pred_mask_class = (pred_mask == class_idx)
    true_mask_class = (true_mask == class_idx)

    intersection = np.logical_and(pred_mask_class, true_mask_class).sum()
    union = np.logical_or(pred_mask_class, true_mask_class).sum()

    if union == 0:
        return float('nan')  # Return NaN if there is no true or predicted pixel
    else:
        return intersection / union

# Compute IoU for all classes
def compute_iou_for_all_classes(model, dataloader, device):
    model.eval()
    all_iou = {class_idx: [] for class_idx in range(NUM_CLASSES)}

    with torch.no_grad():
        for images, true_masks in tqdm(dataloader):
            images = images.to(device)
            true_masks = true_masks.cpu().numpy()

            # Predict masks
            outputs = model(images)
            _, predicted_masks = torch.max(outputs, 1)
            predicted_masks = predicted_masks.cpu().numpy()

            for true_mask, pred_mask in zip(true_masks, predicted_masks):
                for class_idx in range(NUM_CLASSES):
                    iou = calculate_iou(pred_mask, true_mask, class_idx)
                    if not np.isnan(iou):
                        all_iou[class_idx].append(iou)

    avg_iou = {class_idx: np.mean(iou_list) for class_idx, iou_list in all_iou.items()}
    return avg_iou

# Print IoU for each class
def print_iou_per_class(iou_dict):
    for class_idx, iou in iou_dict.items():
        print(f'Class {class_idx} IoU: {iou:.4f}')

# Evaluation script
if __name__ == "__main__":
    # Create dataset and dataloader for validation set
    val_dataset = SegmentationDataset(image_dir=VAL_IMAGE_DIR, image_transform=image_transform, mask_transform=mask_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Initialize and load the model
    model = AttUNet(num_classes=NUM_CLASSES).to(device)
    
    # Load the model with map_location to CPU if GPU is not available
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

    # Compute IoU
    val_iou = compute_iou_for_all_classes(model, val_dataloader, device)

    # Print IoU for each class
    print("IoU for each class on validation set:")
    print_iou_per_class(val_iou)