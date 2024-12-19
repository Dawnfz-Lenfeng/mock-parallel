import os
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from benchmark.benchmark import Benchmark
from config import Config
from models.yolo import myYOLO
from parallel.data_parallel import DataParallel
from parallel.pipeline_parallel import PipelineParallel
from parallel.tensor_parallel import TensorParallel
from utils.dataset import YOLODataset
from utils.utils import detection_collate, gt_creator, compute_loss


def train(model, dataloader, optimizer, device, num_epochs):
    """训练函数"""
    model.train()
    for epoch in range(num_epochs):
        start = time.time()
        train_loss = 0
        for batch_idx, (images, targets) in enumerate(dataloader):
            # 制作训练标签
            targets = [label.tolist() for label in targets]
            targets = gt_creator(
                input_size=Config.IMAGE_SIZE, stride=model.stride, label_lists=targets
            )

            # to device
            images = images.to(device)
            targets = targets.to(device)

            # 前向传播
            conf_pred, cls_pred, txtytwth_pred = model(images)

            # 计算损失
            _, _, _, total_loss = compute_loss(conf_pred, cls_pred, txtytwth_pred, targets)
            train_loss += total_loss.item()

            # 反向传播, 更新梯度
            optimizer.zero_grad()
            total_loss.backward()

            # 更新模型参数
            optimizer.step()

        train_loss /= len(dataloader)
        end = time.time()
        train_time = end - start
        print(
            f"Epoch [{epoch+1}/{num_epochs}], time: {train_time:.4f}s, train_loss: {train_loss:.4f}"
        )


def main():
    # 设置设备
    device = torch.device(Config.DEVICE)

    # 数据预处理
    transform = transforms.Compose(
        [
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )

    # 准备数据集
    image_files = [
        os.path.join(Config.TRAIN_IMAGE_DIR, f)
        for f in os.listdir(Config.TRAIN_IMAGE_DIR)
        if f.endswith(".png")
    ]

    train_dataset = YOLODataset(
        image_files=image_files, labels_dir=Config.TRAIN_LABEL_DIR, transform=transform
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=detection_collate,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 初始化模型
    model = myYOLO().to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 创建不同的并行版本
    '''
    device_ids = [0, 1, 2, 3]  # 假设有4个GPU
    data_parallel_model = DataParallel(model, device_ids)
    tensor_parallel_model = TensorParallel(model, device_ids)
    pipeline_parallel_model = PipelineParallel(model, device_ids, chunks=4)
    '''
    # 训练基础版本
    print("Training base model...")
    train(model, train_dataloader, optimizer, device, Config.NUM_EPOCHS)
    
    '''
    # Benchmark比较
    models = [
        model,
        data_parallel_model,
        tensor_parallel_model,
        pipeline_parallel_model,
    ]
    method_names = ["Base", "DataParallel", "TensorParallel", "PipelineParallel"]

    print("\nRunning benchmarks...")
    results = Benchmark.compare_parallel_methods(
        models, train_dataloader, device, method_names
    )

    # 打印benchmark结果
    for name, metrics in results.items():
        print(f"\n{name} Results:")
        print(f"Average time: {metrics['avg_time']:.4f}s")
        print(f"Min time: {metrics['min_time']:.4f}s")
        print(f"Max time: {metrics['max_time']:.4f}s")

    '''
if __name__ == "__main__":
    main()
