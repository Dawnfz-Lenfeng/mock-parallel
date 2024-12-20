import torch
import torch.nn as nn

from ..models import myYOLO


class PipelineParallel:
    """
    流水线并行接口
    """

    def __init__(self, model: myYOLO, device_ids: list[int], chunks: int):
        """
        Args:
            model: 要并行化的模型
            device_ids: GPU设备ID列表
            chunks: 将输入分成的micro-batch数量
        """
        self.model = model
        self.device_ids = device_ids
        self.chunks = chunks
        self.num_gpus = len(device_ids)

        # 将模型分成几个阶段
        self.stages = [
            nn.Sequential(model.backbone).to(f"cuda:{device_ids[0]}"),
            nn.Sequential(model.neck).to(f"cuda:{device_ids[1]}"),
            nn.Sequential(model.convs).to(f"cuda:{device_ids[2]}"),
            nn.Sequential(model.pred).to(f"cuda:{device_ids[-1]}"),
        ]

    def parallelize(self) -> nn.Module:
        """
        实现流水线并行
        Returns:
            并行化后的模型
        """

        def forward(*inputs):
            x = inputs[0]
            for stage in self.stages:
                x = stage(x)
            return x

        self.model.forward = forward
        return self.model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        流水线并行的前向传播
        """
        # 将输入分成chunks个micro-batch
        micro_batches = x.chunk(self.chunks)

        outputs = []
        for mb in micro_batches:
            # 对每个micro-batch执行流水线并行
            out = mb
            for stage in self.stages:
                out = stage(out)
            outputs.append(out)

        # 合并所有micro-batch的输出
        return torch.cat(outputs, dim=0)
