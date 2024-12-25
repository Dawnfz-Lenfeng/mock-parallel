import torch
import torch.nn as nn
import torch.nn.functional as F

from ..yolo import myYOLO


class ParallelLinear(nn.Module):
    """并行线性层实现"""

    def __init__(self, in_features: int, out_features: int, num_gpus: int):
        super().__init__()
        self.num_gpus = num_gpus
        # 将输出特征分割到不同GPU
        self.out_features_per_gpu = out_features // num_gpus
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.randn(self.out_features_per_gpu, in_features))
                for _ in range(num_gpus)
            ]
        )
        self.biases = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(self.out_features_per_gpu))
                for _ in range(num_gpus)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 在每个GPU上计算部分输出
        outputs = []
        for i in range(self.num_gpus):
            device = f"cuda:{i}"
            local_x = x.to(device)
            local_out = F.linear(local_x, self.weights[i], self.biases[i])
            outputs.append(local_out)
        # 合并所有GPU的结果
        return torch.cat(outputs, dim=1)


class ParallelConv2d(nn.Module):
    """并行卷积层实现"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        num_gpus: int,
    ):
        super().__init__()
        self.num_gpus = num_gpus
        # 将输出通道分割到不同GPU
        self.out_channels_per_gpu = out_channels // num_gpus
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    self.out_channels_per_gpu,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                )
                for _ in range(num_gpus)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 在每个GPU上计算部分输出
        outputs = []
        for i in range(self.num_gpus):
            device = f"cuda:{i}"
            local_x = x.to(device)
            local_out = self.convs[i](local_x)
            outputs.append(local_out)
        # 合并所有GPU的结果
        return torch.cat(outputs, dim=1)


class TensorParallel(myYOLO):
    """张量并行YOLO实现"""

    def __init__(self, model: myYOLO, device_ids: list[int]):
        super().__init__(
            device=f"cuda:{device_ids[0]}",
            input_size=model.input_size,
            num_classes=model.num_classes,
            stride=model.stride,
            conf_thresh=model.conf_thresh,
            nms_thresh=model.nms_thresh,
            trainable=model.trainable,
        )

        self.device_ids = device_ids
        self.num_gpus = len(device_ids)

        # 替换模型中的大型层为并行版本
        self._parallelize_layers()

    def _parallelize_layers(self):
        """将模型中的大型层替换为并行版本"""
        # 替换backbone中的大型卷积层
        for name, module in self.backbone.named_children():
            if isinstance(module, nn.Conv2d) and module.out_channels >= 256:
                setattr(
                    self.backbone,
                    name,
                    ParallelConv2d(
                        module.in_channels,
                        module.out_channels,
                        module.kernel_size[0],
                        module.stride[0],
                        module.padding[0],
                        self.num_gpus,
                    ),
                )

        # 替换neck中的大型卷积层
        for i, module in enumerate(self.neck.children()):
            if isinstance(module, nn.Conv2d) and module.out_channels >= 256:
                self.neck[i] = ParallelConv2d(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size[0],
                    module.stride[0],
                    module.padding[0],
                    self.num_gpus,
                )

        # 替换detection head中的大型卷积层
        for i, module in enumerate(self.convs.children()):
            if isinstance(module, nn.Conv2d) and module.out_channels >= 256:
                self.convs[i] = ParallelConv2d(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size[0],
                    module.stride[0],
                    module.padding[0],
                    self.num_gpus,
                )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        if not self.trainable:
            return self.inference(x)

        # 正常前向传播，并行层会自动处理张量分割和合并
        x = self.backbone(x)
        x = self.neck(x)
        x = self.convs(x)
        x = self.pred(x)

        # 处理输出
        pred = x.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        conf_pred = pred[..., :1]
        cls_pred = pred[..., 1 : 1 + self.num_classes]
        txtytwth_pred = pred[..., 1 + self.num_classes :]

        return conf_pred, cls_pred, txtytwth_pred
