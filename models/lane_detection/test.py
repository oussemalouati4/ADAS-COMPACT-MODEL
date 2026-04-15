"""
Test script for Lane Detection PyTorch model
Load best_model.pth and run inference on images
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path

# ─── Your Model Architecture (copy from training script) ────────────────────

import torch.nn as nn
import torch.nn.functional as F
import timm

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = ConvBNReLU(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class LaneUNet(nn.Module):
    ENCODER_CHANNELS = {
        "efficientnet_b4": [24, 32, 56, 160, 448],
    }

    def __init__(self, encoder_name="efficientnet_b4", num_classes=1):
        super().__init__()
        enc_ch = self.ENCODER_CHANNELS.get(encoder_name, [24, 32, 56, 160, 448])
        
        self._backbone = timm.create_model(encoder_name, pretrained=False, features_only=True)
        self.bridge = ConvBNReLU(enc_ch[4], enc_ch[4] * 2)
        
        d = enc_ch[4] * 2
        self.dec0 = DecoderBlock(d, enc_ch[3], 256)
        self.dec1 = DecoderBlock(256, enc_ch[2], 128)
        self.dec2 = DecoderBlock(128, enc_ch[1], 64)
        self.dec3 = DecoderBlock(64, enc_ch[0], 32)
        self.dec4 = DecoderBlock(32, 0, 16)
        
        self.head = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, 1),
        )

    def forward(self, x):
        features = self._backbone(x)
        s0, s1, s2, s3, s4 = features
        
        x = self.bridge(s4)
        x = self.dec0(x, s3)
        x = self.dec1(x, s2)
        x = self.dec2(x, s1)
        x = self.dec3(x, s0)
        x = self.dec4(x, None)
        
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(x)

# ─── Config ─────────────────────────────────────────────────────────────────

class Config:
    MODEL_PATH = "/kaggle/working/checkpoints/best_model.pth"  # Your model
    IMG_HEIGHT = 384
    IMG_WIDTH = 640
    CONF_THRESHOLD = 0.5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Lane Detector Class ────────────────────────────────────────────────────

class LaneDetector:
    def __init__(self, model_path, device=None):
        self.device = device or Config.DEVICE
        self.model = self._load_model(model_path)
        self.model.eval()
        print(f"✓ Model loaded on {self.device}")
        
    def _load_model(self, path):
        model = LaneUNet(encoder_name="efficientnet_b4", num_classes=1)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
            
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint["model"], strict=False)
        model.to(self.device)
        return model
    
    def preprocess(self, image):
        """Preprocess image for model"""
        original_size = (image.shape[1], image.shape[0])
        
        # Resize
        img_resized = cv2.resize(image, (Config.IMG_WIDTH, Config.IMG_HEIGHT))
        
        # Normalize (ImageNet stats)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_norm = (img_norm - mean) / std
        
        # To tensor (B, C, H, W)
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)
        
        return img_tensor.to(self.device), original_size
    
    def postprocess(self, output, original_size):
        """Convert model output to binary mask"""
        # Sigmoid + threshold
        pred = torch.sigmoid(output)
        mask = (pred > Config.CONF_THRESHOLD).cpu().numpy().astype(np.uint8)
        
        # Resize to original
        mask = mask.squeeze()
        mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
        
        return mask_resized * 255
    
    @torch.no_grad()
    def predict(self, image):
        """Run inference"""
        input_tensor, original_size = self.preprocess(image)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Postprocess
        mask = self.postprocess(output, original_size)
        return mask
    
    def visualize(self, image, mask, alpha=0.6):
        """Create visualization"""
        # Overlay mask on image
        overlay = image.copy()
        overlay[mask > 127] = [0, 0, 255]  # Red lanes
        
        # Blend
        result = cv2.addWeighted(image, 0.4, overlay, 0.6, 0)
        
        # Add metrics text
        lane_ratio = np.sum(mask > 127) / (mask.shape[0] * mask.shape[1]) * 100
        cv2.putText(result, f"Lane: {lane_ratio:.1f}%", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return result

# ─── Test Functions ────────────────────────────────────────────────────────

def test_single_image(detector, image_path, save_path=None):
    """Test on single image and show results"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Cannot load: {image_path}")
        return
    
    # Inference
    mask = detector.predict(image)
    result = detector.visualize(image, mask)
    
    # Save if path provided
    if save_path:
        cv2.imwrite(save_path, result)
        cv2.imwrite(save_path.replace(".jpg", "_mask.jpg"), mask)
        print(f"✓ Saved: {save_path}")
    
    # Display
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Lane Mask")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Prediction")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def test_folder(detector, folder_path, output_folder="test_results"):
    """Test all images in folder"""
    os.makedirs(output_folder, exist_ok=True)
    
    image_paths = glob(os.path.join(folder_path, "*.jpg")) + \
                  glob(os.path.join(folder_path, "*.png"))
    
    print(f"Found {len(image_paths)} images")
    
    for img_path in image_paths:
        filename = Path(img_path).name
        save_path = os.path.join(output_folder, f"pred_{filename}")
        test_single_image(detector, img_path, save_path)
    
    print(f"\n✓ All results saved to: {output_folder}/")

# ─── MAIN ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Initialize detector
    detector = LaneDetector(Config.MODEL_PATH)
    
    # Option 1: Test single image (from validation set)
    val_img = "/kaggle/input/datasets/princekhunt19/road-lane-segmentation-imgs-and-labels/dataset/val/images/i10.jpg"
    if os.path.exists(val_img):
        test_single_image(detector, val_img, "result_val.jpg")
    else:
        print("Val image not found, use your own image path")
    
    # Option 2: Test entire validation folder
    val_folder = "/kaggle/input/datasets/princekhunt19/road-lane-segmentation-imgs-and-labels/dataset/val/images"
    if os.path.exists(val_folder):
        test_folder(detector, val_folder, "val_predictions")
    
    # Option 3: Test your own image
    # test_single_image(detector, "your_image.jpg", "output.jpg")