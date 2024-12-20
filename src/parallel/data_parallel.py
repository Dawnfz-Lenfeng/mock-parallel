import torch
import torch.nn as nn
import torch.nn.parallel as parallel


class DataParallel:
    """
    数据并行接口
    """

    def __init__(self, model: nn.Module, device_ids: list[int]):
        """
        Args:
            model: 要并行化的模型
            device_ids: GPU设备ID列表
        """
        self.model = model
        self.device_ids = device_ids
        self.parallel_model = None

    def parallelize(self) -> nn.Module:
        """
        将模型转换为数据并行模式
        Returns:
            并行化后的模型
        """
        self.parallel_model = parallel.DataParallel(
            self.model, device_ids=self.device_ids
        )
        return self.parallel_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if self.parallel_model is None:
            self.parallelize()
        return self.parallel_model(x)
