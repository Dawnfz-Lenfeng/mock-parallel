import time
from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader

from ..models import myYOLO
from ..parallel import DataParallel, PipelineParallel, TensorParallel


class Benchmark:
    @staticmethod
    def run_benchmark(
        model: Union[myYOLO, DataParallel, TensorParallel, PipelineParallel],
        dataloader: DataLoader,
        device: torch.device,
        num_iterations: int = 100,
    ) -> Dict[str, float]:
        """
        运行基准测试
        Args:
            model: 要测试的模型
            dataloader: 数据加载器
            device: 计算设备
            num_iterations: 测试迭代次数
        Returns:
            包含测试结果的字典
        """
        times: List[float] = []
        model.eval()

        with torch.no_grad():
            for i in range(num_iterations):
                start = time.time()
                for batch in dataloader:
                    images = batch[0].to(device)
                    _ = model(images)
                    torch.cuda.synchronize()  # 确保GPU操作完成
                end = time.time()
                times.append(end - start)

        return {
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "throughput": len(dataloader.dataset)
            / (sum(times) / len(times)),  # 添加吞吐量
        }

    @staticmethod
    def compare_parallel_methods(
        models: List[Union[myYOLO, DataParallel, TensorParallel, PipelineParallel]],
        dataloader: DataLoader,
        device: torch.device,
        method_names: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        比较不同并行方法的性能
        Args:
            models: 要比较的模型列表
            dataloader: 数据加载器
            device: 计算设备
            method_names: 方法名称列表
        Returns:
            包含各方法测试结果的字典
        """
        results = {}
        for model, name in zip(models, method_names):
            print(f"\nTesting {name}...")
            results[name] = Benchmark.run_benchmark(model, dataloader, device)

            # 打印GPU内存使用情况
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.max_memory_allocated() / 1024**2
                memory_reserved = torch.cuda.max_memory_reserved() / 1024**2
                results[name]["memory_allocated"] = memory_allocated
                results[name]["memory_reserved"] = memory_reserved

                # 重置内存统计
                torch.cuda.reset_peak_memory_stats()

        return results
