import torch.nn as nn


class PipelineParallel:
    """
    流水线并行接口
    """

    def __init__(self, model: nn.Module, device_ids, chunks):
        self.model = model
        self.device_ids = device_ids
        self.chunks = chunks

    def parallelize(self):
        """
        实现流水线并行
        """
        pass

    def forward(self, x):
        pass
