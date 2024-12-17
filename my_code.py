import os

import torch
import torch.nn as nn
import torch.optim as optim
from mpi4py import MPI
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

# 初始化 MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def split_dataset(image_files: list[str]) -> list[str]:
    """将数据集按MPI进程数进行划分"""
    subset_size = len(image_files) // size
    start = rank * subset_size
    end = start + subset_size
    return image_files[start:end]


class YOLODataset(torch.utils.data.Dataset):
    """加载YOLO数据集"""

    def __init__(
        self, image_files: list[str], labels_dir: str, S=7, B=2, C=1, transform=None
    ):
        self.image_files = image_files
        self.labels_dir = labels_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # 初始化标签张量
        target = torch.zeros((self.S, self.S, self.B * 5 + self.C))

        # 获取对应标签
        label_path = os.path.join(
            self.labels_dir, img_path.split("/")[-1].replace(".png", ".txt")
        )
        with open(label_path, "r") as f:
            for line in f:
                class_id, x, y, w, h = map(float, line.split())
                class_id = int(class_id)

                # 确定该目标所在的网络
                grid_x = int(x * self.S)
                grid_y = int(y * self.S)

                # 计算相对网络的坐标
                x_cell = x * self.S - grid_x
                y_cell = y * self.S - grid_y

                # 设置边界框
                for b in range(self.B):
                    target[grid_y, grid_x, b * 5] = 1  # confidence
                    target[grid_y, grid_x, b * 5 + 1] = x_cell
                    target[grid_y, grid_x, b * 5 + 2] = y_cell
                    target[grid_y, grid_x, b * 5 + 3] = w
                    target[grid_y, grid_x, b * 5 + 4] = h

                # 设置类别
                target[grid_y, grid_x, self.B * 5 + class_id] = 1

        return img, target


class SimpleYOLO(nn.Module):
    def __init__(self, S=7, B=2, C=1):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C

        # 卷积层
        self.convs = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # (S, S, 16)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (S/2, S/2, 16)

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # (S/2, S/2, 32)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (S/4, S/4, 32)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (S/4, S/4, 64)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (S/8, S/8, 64)
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (224 // 8) ** 2, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, S * S * (B * 5 + C)),
        )

    def forward(self, x: torch.Tensor):
        x = self.convs(x)
        x = self.fc(x)
        x = x.view(-1, self.S, self.S, self.B * 5 + self.C)
        return x


def avg_gradient(model: SimpleYOLO):
    """
    并行最重要的函数! 实现不同进程之间的梯度同步
    每个进程独立计算梯度, 利用MPI将各进程梯度相加, 从而进行平均
    """
    for param in model.parameters():
        if param.grad is not None:
            # 获取当前进程的梯度
            grad = param.grad.data
            grad_copy = grad.clone()
            # 使用Allreduce将所有的梯度相加
            comm.Allreduce(grad, grad_copy, op=MPI.SUM)
            # 写回平均梯度
            param.grad.data.copy_(grad_copy / size)


def train(
    model: SimpleYOLO,
    dataloader: YOLODataset,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 10,
):
    """训练函数"""
    criterion = torch.nn.MSELoss()
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (imgs, targets) in enumerate(dataloader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            # 前向传播
            outputs = model(imgs)

            # 计算损失
            loss = criterion(outputs, targets)

            # 反向传播, 更新梯度
            optimizer.zero_grad()
            loss.backward()

            # 梯度同步
            avg_gradient(model)

            # 更新模型参数
            optimizer.step()

            if rank == 0 and batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item()}"
                )


def main():
    device = torch.device("cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    train_image_dir = "./data/C2A_Dataset/new_dataset3/train/images"
    train_label_dir = "./data/C2A_Dataset/new_dataset3/train/labels"

    # 数据集划分
    all_image_files = [os.path.join(train_image_dir, f) for f in os.listdir(train_image_dir) if f.endswith(".png")]
    # 为了加快推理, 我们只取前100张图片进行训练
    all_image_files = all_image_files[:100]

    subset_image_files = split_dataset(all_image_files)
    dataset = YOLODataset(subset_image_files, train_label_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    # 初始化模型
    model = SimpleYOLO().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.9)

    train(model, dataloader, optimizer, device, num_epochs=10)


if __name__ == "__main__":
    main()
