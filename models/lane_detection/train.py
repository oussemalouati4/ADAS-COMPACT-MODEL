"""
Lane Detection — Full Training Script
UNet with EfficientNet-B4 encoder, Combined BCE+Dice loss,
mixed-precision training, cosine annealing, early stopping.

Requirements:
    pip install torch torchvision timm albumentations tensorboard

Directory layout expected:
    data/
      train/
        images/   *.jpg / *.png
        masks/    *.png  (binary, same stem as image)
      val/
        images/
        masks/
"""

import os
import warnings
import logging

# Suppress OpenCV warnings BEFORE importing cv2
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Also suppress Python warnings
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

import cv2
import numpy as np
import time
from pathlib import Path
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm


# ─── Config ──────────────────────────────────────────────────────────────────

class Config:
    # Use the actual Kaggle dataset path
    DATA_DIR = "/kaggle/input/datasets/princekhunt19/road-lane-segmentation-imgs-and-labels/dataset"
    CHECKPOINT_DIR = "/kaggle/working/checkpoints"
    BEST_MODEL_PATH = "/kaggle/working/checkpoints/best_model.pth"
    LOG_DIR = "/kaggle/working/runs/lane_det"

    # Model
    ENCODER         = "efficientnet_b4"
    PRETRAINED      = True
    NUM_CLASSES     = 1          # binary segmentation

    # Input
    IMG_HEIGHT      = 384
    IMG_WIDTH       = 640

    # Training
    NUM_EPOCHS      = 100
    BATCH_SIZE      = 8
    NUM_WORKERS     = 4
    LEARNING_RATE   = 1e-3
    WEIGHT_DECAY    = 1e-4
    MIXED_PRECISION = True
    DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

    # Loss weights
    BCE_WEIGHT      = 0.5
    DICE_WEIGHT     = 0.5

    # Metrics
    CONF_THRESHOLD  = 0.5

    def make_dirs(self):
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        parent = os.path.dirname(self.BEST_MODEL_PATH)
        if parent:
            os.makedirs(parent, exist_ok=True)


cfg = Config()

def __len__(self):
    return len(self.image_paths)
# ─── Dataset ─────────────────────────────────────────────────────────────────
class LaneDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.image_paths = sorted(
            glob(os.path.join(root, "images", "*.jpg")) +
            glob(os.path.join(root, "images", "*.png"))
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found under {root}/images/")
        
        self.label_dir = os.path.join(root, "labels")  # TXT files here
        self.transform = transform
        
        # Match pairs
        self.pairs = []
        for img_path in self.image_paths:
            stem = Path(img_path).stem
            txt_path = os.path.join(self.label_dir, f"{stem}.txt")
            if os.path.exists(txt_path):
                self.pairs.append((img_path, txt_path))
            else:
                self.pairs.append((img_path, None))

    def __len__(self):
        return len(self.pairs)

    def _txt_to_mask(self, txt_path, img_shape):
        """Convert YOLO format TXT to binary mask"""
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if txt_path is None or not os.path.exists(txt_path):
            return mask
        
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            # YOLO format: class_id x1 y1 x2 y2 ... (normalized 0-1)
            # For lanes, it's usually polygons
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            
            # Convert normalized coords to pixel coords
            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * w)
                y = int(coords[i+1] * h)
                points.append([x, y])
            
            if len(points) >= 2:
                # Draw lane line as thick line or polygon
                points = np.array(points, np.int32)
                cv2.fillPoly(mask, [points], 255)  # Fill polygon
                # Or cv2.polylines(mask, [points], False, 255, thickness=5)
        
        return mask

    def __getitem__(self, idx):
        img_path, txt_path = self.pairs[idx]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert TXT annotation to mask
        mask = self._txt_to_mask(txt_path, image.shape)
        mask = (mask > 127).astype(np.uint8)
        
        if self.transform:
            result = self.transform(image=image, mask=mask)
            image = result["image"]
            mask = result["mask"]
        
        if isinstance(mask, torch.Tensor):
            mask = mask.float().unsqueeze(0)
        else:
            mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        return image, mask
def get_transforms(train: bool):
    h, w = cfg.IMG_HEIGHT, cfg.IMG_WIDTH
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    if train:
        return A.Compose([
            A.Resize(h, w),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=10, p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
                A.CLAHE(p=1),
            ], p=0.4),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(h, w),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


def get_dataloaders():
    train_ds = LaneDataset(
        os.path.join(cfg.DATA_DIR, "train"),
        transform=get_transforms(train=True),
    )
    val_ds = LaneDataset(
        os.path.join(cfg.DATA_DIR, "val"),
        transform=get_transforms(train=False),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )
    print(f"  Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
    return train_loader, val_loader


# ─── Model ───────────────────────────────────────────────────────────────────

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
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = ConvBNReLU(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # Handle size mismatch from rounding
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                                  align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class LaneUNet(nn.Module):
    """
    EfficientNet-B4 encoder + lightweight UNet decoder.
    Encoder feature channels (B4): 24, 32, 56, 160, 448  (indices 0-4)
    """

    ENCODER_CHANNELS = {
        "efficientnet_b0": [16, 24, 40, 112, 320],
        "efficientnet_b1": [16, 24, 40, 112, 320],
        "efficientnet_b2": [16, 24, 48, 120, 352],
        "efficientnet_b3": [24, 32, 48, 136, 384],
        "efficientnet_b4": [24, 32, 56, 160, 448],
        "efficientnet_b5": [24, 40, 64, 176, 512],
    }

    def __init__(self, encoder_name="efficientnet_b4", pretrained=True,
                 num_classes=1):
        super().__init__()
        enc_ch = self.ENCODER_CHANNELS.get(encoder_name, [24, 32, 56, 160, 448])

        # ── Encoder ──────────────────────────────────────────────────────────
        backbone = timm.create_model(
            encoder_name, pretrained=pretrained, features_only=True
        )
        # Expose each stage as a named attribute (for differential LR)
        self.enc0 = backbone.layer0 if hasattr(backbone, "layer0") else backbone.conv_stem
        # timm features_only backbone — we'll just store the whole backbone
        # and call it stage-by-stage via forward_features
        self._backbone = backbone
        self._enc_ch   = enc_ch

        # ── Bridge ───────────────────────────────────────────────────────────
        self.bridge = ConvBNReLU(enc_ch[4], enc_ch[4] * 2)

        # ── Decoder ──────────────────────────────────────────────────────────
        d = enc_ch[4] * 2
        self.dec0 = DecoderBlock(d,       enc_ch[3], 256)
        self.dec1 = DecoderBlock(256,     enc_ch[2], 128)
        self.dec2 = DecoderBlock(128,     enc_ch[1], 64)
        self.dec3 = DecoderBlock(64,      enc_ch[0], 32)
        self.dec4 = DecoderBlock(32,      0,         16)   # no skip at stage 0

        # ── Head ─────────────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, 1),
        )

    # Expose encoder params via named properties (used by train() for diff LR)
    @property
    def enc1(self): return nn.ModuleList([])
    @property
    def enc2(self): return nn.ModuleList([])
    @property
    def enc3(self): return nn.ModuleList([])
    @property
    def enc4(self): return nn.ModuleList([])
    @property
    def enc5(self): return nn.ModuleList([])

    def encoder_params(self):
        return list(self._backbone.parameters())

    def decoder_params(self):
        return (list(self.bridge.parameters()) +
                list(self.dec0.parameters()) +
                list(self.dec1.parameters()) +
                list(self.dec2.parameters()) +
                list(self.dec3.parameters()) +
                list(self.dec4.parameters()) +
                list(self.head.parameters()))

    def forward(self, x):
        features = self._backbone(x)   # list of 5 feature maps
        s0, s1, s2, s3, s4 = features

        x = self.bridge(s4)
        x = self.dec0(x, s3)
        x = self.dec1(x, s2)
        x = self.dec2(x, s1)
        x = self.dec3(x, s0)
        x = self.dec4(x, None)

        # Upsample to input resolution
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(x)


def build_model(pretrained=True):
    return LaneUNet(
        encoder_name=cfg.ENCODER,
        pretrained=pretrained,
        num_classes=cfg.NUM_CLASSES,
    )


# ─── Loss ────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        flat_p = probs.view(probs.size(0), -1)
        flat_t = targets.view(targets.size(0), -1)
        intersection = (flat_p * flat_t).sum(1)
        dice = (2 * intersection + self.smooth) / (
            flat_p.sum(1) + flat_t.sum(1) + self.smooth
        )
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, bce_w=0.5, dice_w=0.5, pos_weight=5.0):
        super().__init__()
        self.bce_w   = bce_w
        self.dice_w  = dice_w
        pw           = torch.tensor([pos_weight])
        self.bce     = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.dice    = DiceLoss()

    def forward(self, logits, targets):
        # Move pos_weight to same device on first use
        if self.bce.pos_weight.device != logits.device:
            self.bce.pos_weight = self.bce.pos_weight.to(logits.device)
        return (self.bce_w * self.bce(logits, targets) +
                self.dice_w * self.dice(logits, targets))


# ─── Metrics ─────────────────────────────────────────────────────────────────

class LaneMetrics:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, logits: torch.Tensor, masks: torch.Tensor):
        preds = (torch.sigmoid(logits) > self.threshold).float()
        t = masks.float()
        self.tp += (preds * t).sum().item()
        self.fp += (preds * (1 - t)).sum().item()
        self.fn += ((1 - preds) * t).sum().item()
        self.tn += ((1 - preds) * (1 - t)).sum().item()

    def compute(self):
        eps = 1e-8
        iou  = self.tp / (self.tp + self.fp + self.fn + eps)
        dice = 2 * self.tp / (2 * self.tp + self.fp + self.fn + eps)
        prec = self.tp / (self.tp + self.fp + eps)
        rec  = self.tp / (self.tp + self.fn + eps)
        return {"iou": iou, "dice": dice, "precision": prec, "recall": rec}


# ─── Checkpoint Helpers ───────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, best_iou, path):
    torch.save({
        "epoch":     epoch,
        "best_iou":  best_iou,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)
    print(f"  ✓ Checkpoint saved → {path}")


def load_checkpoint(model, optimizer, path):
    if not os.path.exists(path):
        return 0, 0.0
    ck = torch.load(path, map_location=cfg.DEVICE)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    print(f"  ✓ Resumed from epoch {ck['epoch']} (best IoU={ck['best_iou']:.4f})")
    return ck["epoch"], ck["best_iou"]


# ─── Epoch Runner ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scaler,
              metrics, device, train: bool):
    model.train() if train else model.eval()
    metrics.reset()
    total_loss = 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for step, (images, masks) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)

            with autocast(enabled=cfg.MIXED_PRECISION):
                logits = model(images)
                # Align logits to mask size if decoder rounding changed dims
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(logits, size=masks.shape[-2:],
                                           mode="bilinear", align_corners=False)
                loss = criterion(logits, masks)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            metrics.update(logits.detach(), masks)

            if step % 50 == 0:
                phase = "TRAIN" if train else "VAL"
                print(f"    [{phase}] step {step}/{len(loader)}"
                      f"  loss={loss.item():.4f}")

    return total_loss / len(loader), metrics.compute()


# ─── Main Train Function ──────────────────────────────────────────────────────

def train():
    cfg.make_dirs()
    device = torch.device(cfg.DEVICE)
    print(f"\n{'='*60}")
    print(f"  Lane Detection Training")
    print(f"  Device: {device} | AMP: {cfg.MIXED_PRECISION}")
    print(f"{'='*60}\n")

    train_loader, val_loader = get_dataloaders()

    model     = build_model(pretrained=cfg.PRETRAINED).to(device)
    criterion = CombinedLoss(cfg.BCE_WEIGHT, cfg.DICE_WEIGHT, pos_weight=5.0)

    # Differential LR: encoder (pretrained) gets 10× lower LR than decoder
    optimizer = optim.AdamW([
        {"params": model.encoder_params(), "lr": cfg.LEARNING_RATE * 0.1},
        {"params": model.decoder_params(), "lr": cfg.LEARNING_RATE},
    ], weight_decay=cfg.WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    scaler              = GradScaler(enabled=cfg.MIXED_PRECISION)
    start_epoch, best_iou = load_checkpoint(model, optimizer, cfg.BEST_MODEL_PATH)
    writer              = SummaryWriter(cfg.LOG_DIR)
    metrics             = LaneMetrics(threshold=cfg.CONF_THRESHOLD)
    patience            = 10
    patience_counter    = 0

    print(f"Starting training for {cfg.NUM_EPOCHS} epochs...\n")

    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        t0 = time.time()
        print(f"\n── Epoch {epoch+1}/{cfg.NUM_EPOCHS}"
              f"  (LR={scheduler.get_last_lr()}) ──")

        train_loss, train_stats = run_epoch(
            model, train_loader, criterion, optimizer,
            scaler, metrics, device, train=True,
        )
        val_loss, val_stats = run_epoch(
            model, val_loader, criterion, optimizer,
            scaler, metrics, device, train=False,
        )

        scheduler.step()
        elapsed = time.time() - t0

        print(f"\n  Train | loss={train_loss:.4f}  IoU={train_stats['iou']:.4f}"
              f"  Dice={train_stats['dice']:.4f}")
        print(f"  Val   | loss={val_loss:.4f}  IoU={val_stats['iou']:.4f}"
              f"  Dice={val_stats['dice']:.4f}  ({elapsed:.1f}s)")

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("IoU",  {"train": train_stats["iou"],
                                    "val":   val_stats["iou"]},   epoch)
        writer.add_scalars("Dice", {"train": train_stats["dice"],
                                    "val":   val_stats["dice"]},  epoch)
        writer.add_scalar("LR",    scheduler.get_last_lr()[0],    epoch)

        val_iou = val_stats["iou"]
        if val_iou > best_iou:
            best_iou         = val_iou
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch + 1,
                            best_iou, cfg.BEST_MODEL_PATH)
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")

        if (epoch + 1) % 10 == 0:
            ck_path = os.path.join(cfg.CHECKPOINT_DIR, f"epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch + 1, best_iou, ck_path)

        if patience_counter >= patience:
            print(f"\n  Early stopping triggered at epoch {epoch+1}")
            break

    writer.close()
    print(f"\n{'='*60}")
    print(f"  Training complete! Best Val IoU: {best_iou:.4f}")
    print(f"  Best model saved to: {cfg.BEST_MODEL_PATH}")
    print(f"{'='*60}\n")


# ▶ RUN TRAINING
if __name__ == "__main__":
    train()