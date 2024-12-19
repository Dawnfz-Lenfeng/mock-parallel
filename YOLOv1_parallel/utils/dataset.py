import os
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class YOLODataset(Dataset):
    """YOLO格式数据集的加载器"""

    def __init__(
        self,
        image_files: List[str],  # 图片文件路径列表
        labels_dir: str,  # 标签文件夹路径
        transform: Optional[transforms.Compose] = None,  # 数据预处理
    ) -> None:
        self.image_files = image_files
        self.labels_dir = labels_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[List[float]]]:
        """
        返回:
            img: 预处理后的图片张量
            target: [[class_id, x, y, w, h], ...] 格式的标签列表
        """
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        target: List[List[float]] = []
        label_path = os.path.join(
            self.labels_dir, img_path.split("/")[-1].replace(".png", ".txt")
        )
        with open(label_path, "r") as f:
            for line in f:
                class_id, x, y, w, h = map(float, line.split())
                target.append([class_id, x, y, w, h])

        return img, target
