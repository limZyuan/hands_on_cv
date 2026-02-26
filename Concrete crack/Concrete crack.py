# ====================================
# Step 1: Setup and Imports
# ====================================

# Update this to point to your dataset folder
dataset_path = './Concrete crack'
images_path = f'{dataset_path}/images'
masks_path = f'{dataset_path}/masks'

# ====================================
# Step 2: Imports
# ====================================
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ====================================
# Step 3: Dataset Class
# ====================================
IMG_HEIGHT, IMG_WIDTH = 128, 128

class CrackDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB").resize((IMG_HEIGHT, IMG_WIDTH))
        mask = Image.open(mask_path).convert("L").resize((IMG_HEIGHT, IMG_WIDTH))

        # Apply transforms.ToTensor correctly
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        return image, mask
    
# ====================================
# Prepare Datasets
# ====================================
dataset = CrackDataset(images_path, masks_path)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")


# ====================================
# Display one batch of data
# ====================================
import matplotlib.pyplot as plt

# Get one batch
images, masks = next(iter(train_loader))

plt.figure(figsize = (12, 6))
for i in range(len(images)):
    plt.subplot(2, len(images), i+1)
    plt.imshow(images[i].permute(1, 2, 0)) # Convert from [C,H,W] to [H,W,C]
    plt.title("Image")
    plt.axis("off")

    plt.subplot(2, len(images), i+1+len(images))
    plt.imshow(masks[i].squeeze(), cmap="gray")
    plt.title("Mask")
    plt.axis("off")

plt.show()

# ====================================
# Step 5: Define UNet Architecture
# ====================================
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace = True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace = True),
            )
        
        self.enc1 = conv_block(3, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.conv_last = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        c1 = self.enc1(x)
        p1 = self.pool(c1)

        c2 = self.enc2(p1)
        p2 = self.pool(c2)

        c3 = self.enc3(p2)
        p3 = self.pool(c3)

        c4 = self.enc4(p3)
        p4 = self.pool(c4)

        bottleneck = self.bottleneck(p4)

        u4 = self.upconv4(bottleneck)
        u4 = torch.cat([u4, c4], dim=1)
        d4 = self.dec4(u4)

        u3 = self.upconv3(d4)
        u3 = torch.cat([u3, c3], dim=1)
        d3 = self.dec3(u3)

        u2 = self.upconv2(d3)
        u2 = torch.cat([u2, c2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.upconv1(d2)
        u1 = torch.cat([u1, c1], dim=1)
        d1 = self.dec1(u1)

        return torch.sigmoid(self.conv_last(d1))
    
# ====================================
# Step 6: Training Setup
# ====================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA Version: {torch.version.cuda}")
model = UNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.00001)


# ====================================
# Step 7: Training Loop
# ====================================
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")


# ====================================
# Step 8: Visualize Random Predictions
# ====================================
import random

model.eval()

# Pick 4 random indices from validation dataset
indices = random.sample(range(len(val_dataset)), 4)

plt.figure(figsize=(12, 8))
for i, idx in enumerate(indices):
    img, mask = val_dataset[idx]
    img, mask = img.unsqueeze(0).to(device), mask.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img)

    # Convert back to numpy for plotting
    img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()
    pred_np = pred.squeeze().cpu().numpy()

    plt.subplot(4, 3, i*3+1)
    plt.imshow(img_np)
    plt.title("Input")
    plt.axis("off")

    plt.subplot(4, 3, i*3+2)
    plt.imshow(mask_np, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(4, 3, i*3+3)
    plt.imshow(pred_np, cmap="gray")
    plt.title("Prediction")
    plt.axis("off")

plt.tight_layout()
plt.show()

torch.save(model.state_dict(), f'{dataset_path}/unet_weights_v2.pth')
