import torch.nn as nn


class DataParallel:
    """
    数据并行接口
    """

    def __init__(self, model: nn.Module, device_ids):
        self.model = model
        self.device_ids = device_ids

    def parallelize(self):
        """
        实现数据并行
        """
        pass

    def forward(self, x):
        pass
