import torch
import torch.nn as nn

from ..yolo import myYOLO


class DataParallel(myYOLO):
    """使用PyTorch内置DataParallel的YOLO实现"""

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
        self.load_state_dict(model.state_dict())

        # 使用PyTorch的DataParallel
        self.backbone = nn.DataParallel(
            self.backbone.to(self.device), device_ids=device_ids
        )
        self.neck = nn.DataParallel(self.neck.to(self.device), device_ids=device_ids)
        self.convs = nn.DataParallel(self.convs.to(self.device), device_ids=device_ids)
        self.pred = nn.DataParallel(self.pred.to(self.device), device_ids=device_ids)


class CustomDataParallel(myYOLO):
    """自定义实现的数据并行YOLO"""

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

        # 在每个GPU上创建模型副本
        self.models = nn.ModuleList()
        for gpu_id in device_ids:
            # 为每个GPU创建完整的模型副本
            device = f"cuda:{gpu_id}"
            model_copy = myYOLO(
                device=device,
                input_size=model.input_size,
                num_classes=model.num_classes,
                stride=model.stride,
                conf_thresh=model.conf_thresh,
                nms_thresh=model.nms_thresh,
                trainable=model.trainable,
            ).to(device)
            # 复制模型参数
            model_copy.load_state_dict(model.state_dict())
            self.models.append(model_copy)

    def scatter_inputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        """将输入数据分散到不同GPU"""
        batch_size = x.size(0)
        chunks = []
        chunk_sizes = [batch_size // self.num_gpus] * self.num_gpus
        # 处理不能整除的情况
        remainder = batch_size % self.num_gpus
        for i in range(remainder):
            chunk_sizes[i] += 1

        start = 0
        for i, size in enumerate(chunk_sizes):
            end = start + size
            chunk = x[start:end].to(f"cuda:{self.device_ids[i]}")
            chunks.append(chunk)
            start = end

        return chunks

    def gather_outputs(
        self, outputs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """收集并合并所有GPU的输出"""
        # 将所有输出移到主GPU并合并
        output = torch.cat([out.to(self.device) for out in outputs], dim=0)

        # 处理输出
        pred = output.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        conf_pred = pred[..., :1]
        cls_pred = pred[..., 1 : 1 + self.num_classes]
        txtytwth_pred = pred[..., 1 + self.num_classes :]

        return conf_pred, cls_pred, txtytwth_pred

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        if not self.trainable:
            return self.inference(x)

        # 将输入分散到不同GPU
        scattered_inputs = self.scatter_inputs(x)

        # 在每个GPU上并行处理
        outputs = []
        for i, chunk in enumerate(scattered_inputs):
            # 使用对应GPU上的模型进行前向传播
            conf_pred, cls_pred, txtytwth_pred = self.models[i](chunk)
            # 将三个输出拼接回原始形式
            B = chunk.size(0)
            H = W = chunk.size(2) // self.stride
            pred = torch.cat([conf_pred, cls_pred, txtytwth_pred], dim=-1)
            pred = pred.view(B, H, W, -1).permute(0, 3, 1, 2)
            outputs.append(pred)

        # 收集并合并输出
        return self.gather_outputs(outputs)
