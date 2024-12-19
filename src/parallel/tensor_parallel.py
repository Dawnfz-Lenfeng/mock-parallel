import torch.nn as nn


class TensorParallel:
    """
    张量并行接口
    """

    def __init__(self, model: nn.Module, device_ids):
        self.model = model
        self.device_ids = device_ids

    def parallelize(self):
        """
        实现张量并行
        """
        pass

    def forward(self, x):
        pass
