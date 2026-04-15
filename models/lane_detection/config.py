import os
import torch

class Config:
    # ─── Dataset Paths ───────────────────────────────────────────────────────────
    # Road Lane Segmentation dataset (Roboflow YOLO segmentation format)
    ROAD_LANE_DIR   = "/kaggle/input/datasets/princekhunt19/road-lane-segmentation-imgs-and-labels/dataset"
    # BDD100K segmentation masks
    BDD100K_SEG_DIR = "/kaggle/input/datasets/solesensei/solesensei_bdd100k/"

    # ─── Image Settings ──────────────────────────────────────────────────────────
    IMG_WIDTH  = 512   # optimal for Jetson Nano: good resolution, fits in 4GB
    IMG_HEIGHT = 256

    # ─── Training ────────────────────────────────────────────────────────────────
    BATCH_SIZE      = 16         # reduce to 8 if OOM
    NUM_EPOCHS      = 60
    LEARNING_RATE   = 1e-3
    WEIGHT_DECAY    = 1e-4
    NUM_WORKERS     = 4
    PIN_MEMORY      = True
    MIXED_PRECISION = True       # AMP — speeds training, reduces VRAM

    # ─── Model ───────────────────────────────────────────────────────────────────
    NUM_CLASSES = 1              # Binary: lane pixel vs background
    PRETRAINED  = True
    DROPOUT     = 0.2

    # ─── Loss weights ────────────────────────────────────────────────────────────
    BCE_WEIGHT  = 0.5
    DICE_WEIGHT = 0.5

    # ─── BDD100K lane class IDs (pixel values in seg PNG masks) ──────────────────
    BDD_LANE_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # ─── Augmentation ────────────────────────────────────────────────────────────
    AUG_HFLIP_P        = 0.5
    AUG_BRIGHTNESS     = 0.3
    AUG_CONTRAST       = 0.3
    AUG_SATURATION     = 0.2
    AUG_HUE            = 0.1
    AUG_MOTION_BLUR_P  = 0.2
    AUG_COARSE_DROPOUT = 0.1

    # ─── Paths ───────────────────────────────────────────────────────────────────
    CHECKPOINT_DIR  = "/kaggle/working/checkpoints"
    BEST_MODEL_PATH = "/kaggle/working/checkpoints/best_model.pth"
    ONNX_PATH       = "/kaggle/working/lane_detection.onnx"
    TRT_ENGINE_PATH = "/kaggle/working/lane_detection_fp16.trt"
    LOG_DIR         = "/kaggle/working/logs"

    # ─── Inference ───────────────────────────────────────────────────────────────
    CONF_THRESHOLD = 0.5

    # ─── Device ──────────────────────────────────────────────────────────────────
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def make_dirs(cls):
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)

cfg = Config()
print(f"Device : {cfg.DEVICE}")
print(f"AMP    : {cfg.MIXED_PRECISION}")
print(f"Image  : {cfg.IMG_WIDTH}×{cfg.IMG_HEIGHT}")