import time
from typing import List

import torch


class Benchmark:
    @staticmethod
    def run_benchmark(model, dataloader, device, num_iterations=100):
        times = []
        model.eval()

        with torch.no_grad():
            for i in range(num_iterations):
                start = time.time()
                for batch in dataloader:
                    images = batch[0].to(device)
                    _ = model(images)
                end = time.time()
                times.append(end - start)

        return {
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
        }

    @staticmethod
    def compare_parallel_methods(
        models: List[torch.nn.Module], dataloader, device, method_names: List[str]
    ):
        results = {}
        for model, name in zip(models, method_names):
            results[name] = Benchmark.run_benchmark(model, dataloader, device)
        return results
