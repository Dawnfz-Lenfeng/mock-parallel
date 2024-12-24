import os

import torch
from src.benchmark import compare_parallel_methods
from src.config import Config
from src.models import YOLODataset, myYOLO
from src.parallel import DataParallel, PipelineParallel, TensorParallel
from src.utils import detection_collate
from torch.utils.data import DataLoader
from torchvision import transforms


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

    # 训练和比较不同的并行版本
    models = {
        "Base": base_model,
        "DataParallel": data_parallel_model,
        "TensorParallel": tensor_parallel_model,
        "PipelineParallel": pipeline_parallel_model,
    }

    # 运行benchmark测试
    print("\nRunning benchmarks...")

    results = compare_parallel_methods(
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
