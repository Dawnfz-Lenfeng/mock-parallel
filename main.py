import os
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.benchmark.benchmark import Benchmark
from src.config import Config
from src.models import YOLODataset, myYOLO
from src.parallel import DataParallel, PipelineParallel, TensorParallel
from src.utils import compute_loss, detection_collate, gt_creator


def train(
    model: myYOLO,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int,
) -> list[float]:
    """
    训练函数
    Args:
        model: 要训练的模型
        dataloader: 数据加载器
        optimizer: 优化器
        device: 计算设备
        num_epochs: 训练轮数
    Returns:
        训练损失列表
    """
    model.train()
    losses: list[float] = []

    for epoch in range(num_epochs):
        start = time.time()
        train_loss = 0.0

        for batch_idx, (images, targets) in enumerate(dataloader):
            # 制作训练标签
            targets = [label.tolist() for label in targets]
            targets = gt_creator(
                input_size=Config.IMAGE_SIZE, stride=model.stride, label_lists=targets
            )

            # 转移到设备
            images = images.to(device)
            targets = targets.to(device)

            # 前向传播
            conf_pred, cls_pred, txtytwth_pred = model(images)

            # 计算损失
            total_loss = compute_loss(conf_pred, cls_pred, txtytwth_pred, targets)
            train_loss += total_loss.item()

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:  # 每10个batch打印一次
                print(
                    f"Batch [{batch_idx}/{len(dataloader)}], Loss: {total_loss.item():.4f}"
                )

        # 计算epoch平均损失
        train_loss /= len(dataloader)
        losses.append(train_loss)

        # 打印训练信息
        end = time.time()
        train_time = end - start
        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Time: {train_time:.2f}s, "
            f"Loss: {train_loss:.4f}"
        )

        # 打印GPU内存使用情况
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.max_memory_allocated() / 1024**2
            memory_reserved = torch.cuda.max_memory_reserved() / 1024**2
            print(
                f"GPU Memory: Allocated: {memory_allocated:.1f}MB, "
                f"Reserved: {memory_reserved:.1f}MB"
            )

    return losses


def create_model_copy(model: myYOLO, device: str) -> myYOLO:
    """创建模型的深度复制"""
    new_model = myYOLO(
        device=device,
        input_size=Config.IMAGE_SIZE,
        num_classes=Config.NUM_CLASSES,
        stride=Config.STRIDE,
        trainable=True,
    ).to(device)

    # 复制模型参数
    new_model.load_state_dict(model.state_dict())
    return new_model


def main():
    # 设置设备
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    # 获取可用的GPU数量
    num_gpus = torch.cuda.device_count()
    device_ids = list(range(num_gpus))
    print(f"Available GPUs: {num_gpus}")

    # 初始化基础模型
    base_device = f"cuda:{device_ids[0]}"
    base_model = myYOLO(
        device=base_device,
        input_size=Config.IMAGE_SIZE,
        num_classes=Config.NUM_CLASSES,
        stride=Config.STRIDE,
        trainable=True,
    ).to(base_device)

    # 创建不同的并行版本
    data_parallel_model = DataParallel(base_model, device_ids)
    tensor_parallel_model = TensorParallel(base_model, device_ids)
    pipeline_parallel_model = PipelineParallel(base_model, device_ids, chunks=4)

    # 为每个模型创建独立的优化器
    optimizers = {
        "Base": optim.Adam(base_model.parameters(), lr=Config.LEARNING_RATE),
        "DataParallel": optim.Adam(
            data_parallel_model.parameters(), lr=Config.LEARNING_RATE
        ),
        "TensorParallel": optim.Adam(
            tensor_parallel_model.parameters(), lr=Config.LEARNING_RATE
        ),
        "PipelineParallel": optim.Adam(
            pipeline_parallel_model.parameters(), lr=Config.LEARNING_RATE
        ),
    }

    # 训练和比较不同的并行版本
    models = {
        "Base": base_model,
        "DataParallel": data_parallel_model,
        "TensorParallel": tensor_parallel_model,
        "PipelineParallel": pipeline_parallel_model,
    }

    # 训练每个模型
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        # 确保模型在训练模式
        model.train()
        model.trainable = True
        train(model, train_dataloader, optimizers[name], device, Config.NUM_EPOCHS)

    # 运行benchmark测试
    print("\nRunning benchmarks...")
    # 确保所有模型都在评估模式
    for model in models.values():
        model.eval()
        model.trainable = False

    results = Benchmark.compare_parallel_methods(
        list(models.values()), train_dataloader, device, list(models.keys())
    )

    # 打印benchmark结果
    for name, metrics in results.items():
        print(f"\n{name} Results:")
        print(f"Average time: {metrics['avg_time']:.4f}s")
        print(f"Min time: {metrics['min_time']:.4f}s")
        print(f"Max time: {metrics['max_time']:.4f}s")
        print(f"Throughput: {metrics['throughput']:.2f} samples/s")
        if "memory_allocated" in metrics:
            print(f"GPU Memory Allocated: {metrics['memory_allocated']:.1f}MB")
            print(f"GPU Memory Reserved: {metrics['memory_reserved']:.1f}MB")


if __name__ == "__main__":
    main()
