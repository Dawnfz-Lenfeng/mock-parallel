from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Data settings
    TRAIN_IMAGE_DIR: str = "/mnt/C2A_Dataset/new_dataset3/train/images"
    TRAIN_LABEL_DIR: str = "/mnt/C2A_Dataset/new_dataset3/train/labels"
    IMAGE_SIZE: int = 224
    BATCH_SIZE: int = 64
    NUM_WORKERS: int = 4

    # Training settings
    LEARNING_RATE: float = 1e-3
    NUM_EPOCHS: int = 1
    DEVICE: str = "cuda"

    # Model settings
    NUM_CLASSES: int = 5
    STRIDE: int = 32
    CONF_THRESH: float = 0.05
    NMS_THRESH: float = 0.5
    TRAINABLE: bool = True
