import os
import cv2
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─── Augmentation Pipelines ───────────────────────────────────────────────────

def get_train_transforms():
    return A.Compose([
        A.Resize(cfg.IMG_HEIGHT, cfg.IMG_WIDTH),
        A.HorizontalFlip(p=cfg.AUG_HFLIP_P),
        A.ColorJitter(
            brightness=cfg.AUG_BRIGHTNESS, contrast=cfg.AUG_CONTRAST,
            saturation=cfg.AUG_SATURATION, hue=cfg.AUG_HUE, p=0.8,
        ),
        A.MotionBlur(blur_limit=5, p=cfg.AUG_MOTION_BLUR_P),
        A.CoarseDropout(
            max_holes=8, max_height=16, max_width=16,
            fill_value=0, mask_fill_value=0, p=cfg.AUG_COARSE_DROPOUT,
        ),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(cfg.IMG_HEIGHT, cfg.IMG_WIDTH),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ─── Road Lane Dataset (YOLO segmentation format) ─────────────────────────────

def yolo_polygon_to_mask(label_path: str, img_h: int, img_w: int) -> np.ndarray:
    """Convert YOLO segmentation .txt (normalized polygons) to binary mask."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not os.path.exists(label_path):
        return mask
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            coords = list(map(float, parts[1:]))
            points = []
            for i in range(0, len(coords) - 1, 2):
                x = int(coords[i]     * img_w)
                y = int(coords[i + 1] * img_h)
                points.append([x, y])
            if len(points) >= 3:
                pts = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
    return mask


class RoadLaneDataset(Dataset):
    """Roboflow Road Lane Segmentation dataset (YOLO polygon format)."""

    def __init__(self, split: str = "train", transform=None):
        assert split in ("train", "val", "test")
        self.transform = transform
        self.img_dir   = Path(cfg.ROAD_LANE_DIR) / split / "images"
        self.lbl_dir   = Path(cfg.ROAD_LANE_DIR) / split / "labels"
        self.images    = sorted([
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])
        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")
        print(f"[RoadLane] {split}: {len(self.images)} samples")

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        lbl_path = self.lbl_dir / (img_path.stem + ".txt")
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        h, w  = image.shape[:2]
        mask  = (yolo_polygon_to_mask(str(lbl_path), h, w) > 127).astype(np.float32)
        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask  = aug["mask"].unsqueeze(0)
        return image, mask


# ─── BDD100K Dataset (PNG mask format) ────────────────────────────────────────

BDD_LANE_CLASS_IDS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

class BDD100KLaneDataset(Dataset):
    """BDD100K segmentation dataset — extracts lane-marking pixels."""

    def __init__(self, split: str = "train", transform=None):
        assert split in ("train", "val")
        self.transform = transform
        self.img_dir   = Path(cfg.BDD100K_SEG_DIR) / "images" / split
        self.lbl_dir   = Path(cfg.BDD100K_SEG_DIR) / "labels" / split
        self.images    = sorted([
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])
        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")
        print(f"[BDD100K] {split}: {len(self.images)} samples")

    def __len__(self): return len(self.images)

    def _load_mask(self, img_stem: str) -> np.ndarray:
        for ext in (".png", ".jpg"):
            p = self.lbl_dir / (img_stem + ext)
            if p.exists():
                mask_raw = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if mask_raw is not None:
                    binary = np.zeros_like(mask_raw, dtype=np.float32)
                    for cid in BDD_LANE_CLASS_IDS:
                        binary[mask_raw == cid] = 1.0
                    return binary
        return np.zeros((720, 1280), dtype=np.float32)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image    = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask     = self._load_mask(img_path.stem)
        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask  = aug["mask"].unsqueeze(0)
        return image, mask


# ─── Combined DataLoaders ─────────────────────────────────────────────────────

def get_dataloaders():
    train_tf = get_train_transforms()
    val_tf   = get_val_transforms()

    rl_train  = RoadLaneDataset("train", train_tf)
    rl_val    = RoadLaneDataset("val",   val_tf)
    bdd_train = BDD100KLaneDataset("train", train_tf)
    bdd_val   = BDD100KLaneDataset("val",   val_tf)

    train_ds  = ConcatDataset([rl_train, bdd_train])
    val_ds    = ConcatDataset([rl_val,   bdd_val])
    print(f"\n[Dataset] Total train: {len(train_ds)} | val: {len(val_ds)}\n")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY,
    )
    return train_loader, val_loader

print("✓ Dataset classes defined")