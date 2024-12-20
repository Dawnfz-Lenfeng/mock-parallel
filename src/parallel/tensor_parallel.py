import torch
import torch.nn as nn

from ..models import myYOLO


class TensorParallel:
    """
    张量并行接口
    """

    def __init__(self, model: myYOLO, device_ids: list[int]):
        self.model = model
        self.device_ids = device_ids
        self.num_gpus = len(device_ids)

        # 记录需要进行张量并行的层
        self.parallel_layers = {
            "backbone": model.backbone,
            "neck": model.neck,
            "convs": model.convs,
            "pred": model.pred,
        }

    def parallelize(self) -> nn.Module:
        """
        实现张量并行：将每层的通道分割到不同GPU上
        Returns:
            并行化后的模型
        """
        for name, layer in self.parallel_layers.items():
            if isinstance(layer, nn.Sequential):
                # 对Sequential中的每一层进行通道分割
                for i, sublayer in enumerate(layer):
                    if isinstance(sublayer, nn.Conv2d):
                        out_channels = sublayer.out_channels
                        channels_per_gpu = out_channels // self.num_gpus

                        # 分割输出通道
                        splits = []
                        for gpu_id in self.device_ids:
                            split = nn.Conv2d(
                                sublayer.in_channels,
                                channels_per_gpu,
                                sublayer.kernel_size,
                                sublayer.stride,
                                sublayer.padding,
                            ).to(f"cuda:{gpu_id}")
                            splits.append(split)

                        # 替换原来的层
                        layer[i] = nn.ModuleList(splits)

            elif isinstance(layer, nn.Conv2d):
                # 直接分割单个卷积层
                out_channels = layer.out_channels
                channels_per_gpu = out_channels // self.num_gpus

                splits = []
                for gpu_id in self.device_ids:
                    split = nn.Conv2d(
                        layer.in_channels,
                        channels_per_gpu,
                        layer.kernel_size,
                        layer.stride,
                        layer.padding,
                    ).to(f"cuda:{gpu_id}")
                    splits.append(split)

                # 替换原来的层
                setattr(self.model, name, nn.ModuleList(splits))

        return self.model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        张量并行的前向传播
        """
        outputs = []

        # 在每个GPU上并行计算
        for i, gpu_id in enumerate(self.device_ids):
            x_gpu = x.to(f"cuda:{gpu_id}")
            out = self.model(x_gpu)
            outputs.append(out)

        # 合并所有GPU的输出
        return torch.cat(outputs, dim=1)
