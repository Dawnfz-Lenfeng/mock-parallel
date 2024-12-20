import torch
import torch.nn as nn

from ..models import myYOLO


class DataParallel(myYOLO):
    """数据并行YOLO实现"""

    def __init__(self, model: myYOLO, device_ids: list[int]):
        # 继承父类的所有参数
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
        # 加载原始模型的参数
        self.load_state_dict(model.state_dict())

        # 将模型转换为DataParallel并分配到指定设备
        self.backbone = nn.DataParallel(
            self.backbone.to(self.device), device_ids=device_ids
        )
        self.neck = nn.DataParallel(self.neck.to(self.device), device_ids=device_ids)
        self.convs = nn.DataParallel(self.convs.to(self.device), device_ids=device_ids)
        self.pred = nn.DataParallel(self.pred.to(self.device), device_ids=device_ids)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        if not self.trainable:
            return self.inference(x)

        # 输入数据移到主设备
        x = x.to(self.device)

        # 前向传播
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
