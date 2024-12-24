import time

import torch
from torch.utils.data import DataLoader

from .config import Config
from .models import myYOLO
from .utils import compute_loss, gt_creator


def run_benchmark(
    model: myYOLO,
    dataloader: DataLoader,
    device: torch.device,
    num_iterations: int = 1,
) -> dict[str, float]:
    """
    运行训练性能基准测试
    Args:
        model: 要测试的模型
        dataloader: 数据加载器
        device: 计算设备
        num_iterations: 测试迭代次数
    Returns:
        包含测试结果的字典
    """
    times: list[float] = []
    training_losses: list[float] = []
    all_batch_times: list[float] = []

    # 设置为训练模式
    model.train()
    model.trainable = True
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    for i in range(num_iterations):
        batch_times: list[float] = []
        epoch_loss = 0.0

        for batch_idx, (images, targets) in enumerate(dataloader):
            # 准备训练数据
            targets = [label.tolist() for label in targets]
            targets = gt_creator(
                input_size=224, stride=model.stride, label_lists=targets
            )
            images, targets = images.to(device), targets.to(device)

            # 记录开始时间
            start = time.time()

            # 训练步骤
            optimizer.zero_grad()
            conf_pred, cls_pred, txtytwth_pred = model(images)
            loss = compute_loss(conf_pred, cls_pred, txtytwth_pred, targets)
            loss.backward()
            optimizer.step()

            # 确保GPU操作完成
            torch.cuda.synchronize()

            # 记录结束时间
            end = time.time()
            batch_time = end - start
            batch_times.append(batch_time)
            all_batch_times.append(batch_time)
            epoch_loss += loss.item()
            if batch_idx % 10 == 0:  # 每10个batch打印一次
                print(f"Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        times.append(sum(batch_times))
        training_losses.append(epoch_loss / len(dataloader))

        print(f"\rIteration {i+1}/{num_iterations}", end="")

    # 收集GPU内存统计
    memory_stats = {
        "memory_allocated": torch.cuda.max_memory_allocated() / 1024**2,  # MB
        "memory_reserved": torch.cuda.max_memory_reserved() / 1024**2,  # MB
    }

    return {
        "avg_epoch_time": sum(times) / len(times),
        "min_epoch_time": min(times),
        "max_epoch_time": max(times),
        "avg_batch_time": sum(all_batch_times) / len(all_batch_times),
        "samples_per_second": len(dataloader.dataset) / (sum(times) / len(times)),
        "avg_loss": sum(training_losses) / len(training_losses),
        **memory_stats,
    }


def compare_parallel_methods(
    models: list[myYOLO],
    dataloader: DataLoader,
    device: torch.device,
    method_names: list[str],
) -> dict[str, dict[str, float]]:
    """
    比较不同并行方法的训练性能
    """
    results = {}
    for model, name in zip(models, method_names):
        print(f"\nTesting {name}...")
        results[name] = run_benchmark(model, dataloader, device)

        # 重置GPU内存统计
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    return results
