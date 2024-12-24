import copy

import torch
import torch.nn as nn

from ..models import myYOLO


class TensorParallel(myYOLO):
    """张量并行YOLO实现"""

    def __init__(self, model: myYOLO, device_ids: list[int]):
        super().__init__(
            device=f"cuda:{device_ids[0]}",  # 主设备设为第一个GPU
            input_size=model.input_size,
            num_classes=model.num_classes,
            stride=model.stride,
            conf_thresh=model.conf_thresh,
            nms_thresh=model.nms_thresh,
            trainable=model.trainable,
        )

        self.device_ids = device_ids
        self.num_gpus = len(device_ids)

        # 将每个子模块分配到不同GPU
        self._distribute_modules()

    def _distribute_modules(self):
        """将模型的不同部分分配到不同GPU"""
        # 为每个GPU创建一个完整的backbone副本
        backbones = []
        for gpu_id in self.device_ids:
            device = f"cuda:{gpu_id}"
            backbone_copy = copy.deepcopy(self.backbone)
            backbone_copy = backbone_copy.to(device)
            backbones.append(backbone_copy)
        self.backbone = nn.ModuleList(backbones)

        # 为每个GPU创建neck的一部分
        neck_layers = list(self.neck.children())
        neck_per_gpu = max(len(neck_layers) // self.num_gpus, 1)
        neck_splits = []

        for i, gpu_id in enumerate(self.device_ids):
            device = f"cuda:{gpu_id}"
            start_idx = i * neck_per_gpu
            end_idx = (
                start_idx + neck_per_gpu if i < self.num_gpus - 1 else len(neck_layers)
            )
            if start_idx < len(neck_layers):
                split = nn.Sequential(*neck_layers[start_idx:end_idx]).to(device)
                neck_splits.append(split)
        self.neck = nn.ModuleList(neck_splits)

        # 为每个GPU创建convs的一部分
        convs_layers = list(self.convs.children())
        convs_per_gpu = max(len(convs_layers) // self.num_gpus, 1)
        convs_splits = []

        for i, gpu_id in enumerate(self.device_ids):
            device = f"cuda:{gpu_id}"
            start_idx = i * convs_per_gpu
            end_idx = (
                start_idx + convs_per_gpu
                if i < self.num_gpus - 1
                else len(convs_layers)
            )
            if start_idx < len(convs_layers):
                split = nn.Sequential(*convs_layers[start_idx:end_idx]).to(device)
                convs_splits.append(split)
        self.convs = nn.ModuleList(convs_splits)

        # pred层分配到最后一个GPU
        self.pred = self.pred.to(f"cuda:{self.device_ids[-1]}")

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        if not self.trainable:
            return self.inference(x)

        # 输入数据分割并移到不同GPU
        batch_size = x.size(0)
        chunk_size = max(batch_size // self.num_gpus, 1)
        chunks = []
        for i, gpu_id in enumerate(self.device_ids):
            device = f"cuda:{gpu_id}"
            if i == self.num_gpus - 1:
                chunk = x[i * chunk_size :].to(device)
            else:
                chunk = x[i * chunk_size : (i + 1) * chunk_size].to(device)
            chunks.append(chunk)

        # 在不同GPU上并行处理
        outputs = []
        for i, chunk in enumerate(chunks):
            # backbone - 每个GPU都有完整的backbone
            feat = self.backbone[i](chunk)

            # neck - 串行处理每个部分
            for j in range(len(self.neck)):
                device = f"cuda:{self.device_ids[j]}"
                feat = feat.to(device)
                feat = self.neck[j](feat)

            # convs - 串行处理每个部分
            for j in range(len(self.convs)):
                device = f"cuda:{self.device_ids[j]}"
                feat = feat.to(device)
                feat = self.convs[j](feat)

            # pred - 在最后一个GPU上处理
            feat = feat.to(f"cuda:{self.device_ids[-1]}")
            pred = self.pred(feat)
            outputs.append(pred)

        # 在最后一个GPU上合并结果
        output = torch.cat(outputs, dim=0)

        # 处理输出
        pred = output.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        conf_pred = pred[..., :1]
        cls_pred = pred[..., 1 : 1 + self.num_classes]
        txtytwth_pred = pred[..., 1 + self.num_classes :]

        # 确保所有输出都在主设备上
        main_device = f"cuda:{self.device_ids[0]}"
        conf_pred = conf_pred.to(main_device)
        cls_pred = cls_pred.to(main_device)
        txtytwth_pred = txtytwth_pred.to(main_device)

        return conf_pred, cls_pred, txtytwth_pred
