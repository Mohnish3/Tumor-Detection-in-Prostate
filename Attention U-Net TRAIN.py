import sys
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from torch.utils.tensorboard import SummaryWriter
import cv2

# Determine the base directory
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
elif (__file__ is not None) and (os.path.dirname(__file__) != ''):
    BASE_DIR = os.path.dirname(__file__)
else:
    BASE_DIR = os.getcwd()

print("Base Dir:", BASE_DIR)
os.chdir(BASE_DIR)

# Configuration Constants
TRAIN_IMAGE_DIR = 'Raw datasets/Segmentation/Training'
VAL_IMAGE_DIR = 'Raw datasets/Segmentation/Validation'
IMAGE_HEIGHT = 1024  # 128,256,512,1024
IMAGE_WIDTH = 1024  # 128,258,512,1024
BATCH_SIZE = 4
NUM_CLASSES = 4
LEARNING_RATE = 1e-3
NUM_EPOCHS = 2000
AUGMENTATION = True
MODEL_SAVE_PATH = 'models/' + str(IMAGE_HEIGHT) + '_X_' + str(IMAGE_WIDTH) +'_'+str(AUGMENTATION)+'_'+ '_att_unet_model'
TENSORBOARD_LOG_DIR = 'tensorboard_logs_attention'

# Ensure the tensorboard_logs directory exists
if not os.path.exists(TENSORBOARD_LOG_DIR):
    os.makedirs(TENSORBOARD_LOG_DIR)

# Map colors to class indices
color_to_class = {
    (0, 0, 0): 0,  # Background (Black)
    (0, 0, 255): 1,  # Stroma (Blue)
    (0, 255, 0): 2,  # Benign (Green)
    (255, 255, 0): 3  # Tumor (Yellow)
}

class_to_color = {v: k for k, v in color_to_class.items()}


# Dataset Class
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, image_transform=None, mask_transform=None, apply_augmentation=False):
        self.image_dir = image_dir
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.png') and not f.endswith('_mask.png')]
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.apply_augmentation = apply_augmentation

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.image_dir, img_name.replace('.png', '_mask.png'))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        if self.apply_augmentation:
            image, mask = self.augment(image, mask)

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = self.to_class_indices(mask)
        return image, mask

    def augment(self, image, mask):
        # Apply the same transformation to both the image and the mask
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        if random.random() > 0.5:
            angle = random.randint(-180, 180)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
        if random.random() > 0.5:
            image = TF.gaussian_blur(image, kernel_size=(5, 9), sigma=(0.1, 5))
            mask = TF.gaussian_blur(mask, kernel_size=(5, 9), sigma=(0.1, 5))
        if random.random() > 0.5:
            shear = random.uniform(-20, 20)
            image = TF.affine(image, angle=0, translate=(0, 0), scale=1.0, shear=shear)
            mask = TF.affine(mask, angle=0, translate=(0, 0), scale=1.0, shear=shear)

        return image, mask

    def to_class_indices(self, mask):
        mask = np.array(mask)
        class_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
        for rgb, class_idx in color_to_class.items():
            indices = np.all(mask == rgb, axis=-1)
            class_mask[indices] = class_idx
        return torch.from_numpy(class_mask)


# Define transformations for images and masks
image_transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor()
])
mask_transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=transforms.InterpolationMode.NEAREST),
])

# Augmentations (if AUGMENTATION is True)
if AUGMENTATION:
    AUGMENT_PROBABILITY = {
        'hflip': 0.5,
        'vflip': 0.5,
        'rotation': 0.5,
        'blur': 0.3,
        'shear': 0.4
    }

    class RandomApplyTransform:
        def __init__(self, transform, probability):
            self.transform = transform
            self.probability = probability

        def __call__(self, img):
            if random.random() < self.probability:
                return self.transform(img)
            return img

    # Define shear operation
    shear_transform = transforms.RandomAffine(degrees=0, shear=20)

    train_image_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        RandomApplyTransform(TF.hflip, AUGMENT_PROBABILITY['hflip']),
        RandomApplyTransform(TF.vflip, AUGMENT_PROBABILITY['vflip']),
        RandomApplyTransform(transforms.RandomRotation(degrees=(-180, 180)), AUGMENT_PROBABILITY['rotation']),
        RandomApplyTransform(transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), AUGMENT_PROBABILITY['blur']),
        RandomApplyTransform(shear_transform, AUGMENT_PROBABILITY['shear']),
        transforms.ToTensor(),
    ])

    train_mask_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=transforms.InterpolationMode.NEAREST),
        RandomApplyTransform(TF.hflip, AUGMENT_PROBABILITY['hflip']),
        RandomApplyTransform(TF.vflip, AUGMENT_PROBABILITY['vflip']),
        RandomApplyTransform(transforms.RandomRotation(degrees=(-90, 90)), AUGMENT_PROBABILITY['rotation']),
        RandomApplyTransform(shear_transform, AUGMENT_PROBABILITY['shear']),
    ])
else:
    train_image_transform = image_transform
    train_mask_transform = mask_transform

# Create Dataset and DataLoader for training
train_dataset = SegmentationDataset(image_dir=TRAIN_IMAGE_DIR, image_transform=train_image_transform, mask_transform=train_mask_transform, apply_augmentation=AUGMENTATION)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Create Dataset and DataLoader for validation
val_dataset = SegmentationDataset(image_dir=VAL_IMAGE_DIR, image_transform=image_transform, mask_transform=mask_transform)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


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


def apply_color_overlay(image, mask, class_to_color, alpha=0.5):
    color_overlay = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    for class_idx, color in class_to_color.items():
        binary_mask = (mask == class_idx).numpy().astype(np.uint8)
        colored_mask = np.stack([binary_mask * c for c in color], axis=-1)
        color_overlay = np.maximum(color_overlay, colored_mask)

    image_with_overlay = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    overlayed_image = cv2.addWeighted(image_with_overlay, 1 - alpha, color_overlay, alpha, 0)
    return overlayed_image



# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device = ', device)
model = AttUNet(num_classes=NUM_CLASSES).to(device)


# Combined Loss (CrossEntropy + Dice Loss)
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def soft_dice_loss(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        encoded_targets = nn.functional.one_hot(targets.long(), num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()
        intersection = torch.sum(probs * encoded_targets, dim=(0, 2, 3))
        union = torch.sum(probs, dim=(0, 2, 3)) + torch.sum(encoded_targets, dim=(0, 2, 3))
        dice_loss = 1 - 2 * intersection / union
        return dice_loss.mean()

    def forward(self, logits, targets):
        ce_loss = self.ce_loss(logits, targets)
        dice_loss = self.soft_dice_loss(logits, targets)
        return ce_loss + dice_loss, ce_loss, dice_loss


criterion = CombinedLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# TensorBoard Summary Writer
writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)

best_val_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    # Training Phase
    model.train()
    running_loss = 0.0
    running_ce_loss = 0.0
    running_dice_loss = 0.0
    for images, masks in train_dataloader:
        images = images.to(device)
        masks = masks.to(device).long()

        # Forward pass
        outputs = model(images)
        loss, ce_loss, dice_loss = criterion(outputs, masks)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_ce_loss += ce_loss.item()
        running_dice_loss += dice_loss.item()

    epoch_loss = running_loss / len(train_dataloader)
    epoch_ce_loss = running_ce_loss / len(train_dataloader)
    epoch_dice_loss = running_dice_loss / len(train_dataloader)
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Training Loss: {epoch_loss}, CE Loss: {epoch_ce_loss}, Dice Loss: {epoch_dice_loss}')

    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('CE_Loss/train', epoch_ce_loss, epoch)
    writer.add_scalar('Dice_Loss/train', epoch_dice_loss, epoch)

    # Validation Phase
    model.eval()
    val_loss = 0.0
    val_ce_loss = 0.0
    val_dice_loss = 0.0
    with torch.no_grad():
        for images, masks in val_dataloader:
            images = images.to(device)
            masks = masks.to(device).long()

            # Forward pass
            outputs = model(images)
            loss, ce_loss, dice_loss = criterion(outputs, masks)

            val_loss += loss.item()
            val_ce_loss += ce_loss.item()
            val_dice_loss += dice_loss.item()

    val_epoch_loss = val_loss / len(val_dataloader)
    val_epoch_ce_loss = val_ce_loss / len(val_dataloader)
    val_epoch_dice_loss = val_dice_loss / len(val_dataloader)
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {val_epoch_loss}, CE Loss: {val_epoch_ce_loss}, Dice Loss: {val_epoch_dice_loss}')

    writer.add_scalar('Loss/val', val_epoch_loss, epoch)
    writer.add_scalar('CE_Loss/val', val_epoch_ce_loss, epoch)
    writer.add_scalar('Dice_Loss/val', val_epoch_dice_loss, epoch)

    # Save the best model with suffix as the loss
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        model_save_path_with_suffix = f"{MODEL_SAVE_PATH}_val_loss_{val_epoch_loss:.4f}.pth"
        torch.save(model.state_dict(), model_save_path_with_suffix)
        print(f'New best model saved at {model_save_path_with_suffix} with validation loss: {val_epoch_loss}')

print('Training and Validation Finished')
writer.close()