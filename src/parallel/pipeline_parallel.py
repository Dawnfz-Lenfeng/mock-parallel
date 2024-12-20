from typing import Tuple

import torch
import torch.nn as nn

from ..models import myYOLO


class PipelineParallel(myYOLO):
    """流水线并行YOLO实现"""

    def __init__(self, model: myYOLO, device_ids: list[int], chunks: int):
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
        self.chunks = chunks
        self.num_gpus = len(device_ids)
        # 加载原始模型的参数
        self.load_state_dict(model.state_dict())

        # 将不同阶段分配到不同GPU
        self._distribute_stages()

    def _distribute_stages(self):
        """将模型的不同阶段分配到不同GPU"""
        # backbone -> GPU 0
        self.backbone = self.backbone.to(f"cuda:{self.device_ids[0]}")
        # neck -> GPU 1
        self.neck = self.neck.to(f"cuda:{self.device_ids[1 % self.num_gpus]}")
        # convs -> GPU 2
        self.convs = self.convs.to(f"cuda:{self.device_ids[2 % self.num_gpus]}")
        # pred -> GPU 3
        self.pred = self.pred.to(f"cuda:{self.device_ids[3 % self.num_gpus]}")

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        if not self.trainable:
            return self.inference(x)

        batch_size = x.size(0)
        # 确保micro-batch大小能整除batch_size
        self.chunks = min(self.chunks, batch_size)
        while batch_size % self.chunks != 0:
            self.chunks -= 1

        # 将输入数据分割成多个micro-batch
        micro_batches = x.chunk(self.chunks)
        outputs = []

        # 使用流水线方式进行前向传播
        for i in range(self.chunks):
            # 在第一个GPU上处理新的micro-batch（backbone阶段）
            curr_mb = micro_batches[i].to(f"cuda:{self.device_ids[0]}")
            feat = self.backbone(curr_mb)

            # 在第二个GPU上处理neck阶段
            feat = feat.to(f"cuda:{self.device_ids[1 % self.num_gpus]}")
            feat = self.neck(feat)

            # 在第三个GPU上处理convs阶段
            feat = feat.to(f"cuda:{self.device_ids[2 % self.num_gpus]}")
            feat = self.convs(feat)

            # 在第四个GPU上处理pred阶段
            feat = feat.to(f"cuda:{self.device_ids[3 % self.num_gpus]}")
            pred = self.pred(feat)
            outputs.append(pred)

        # 合并所有micro-batch的输出
        output = torch.cat(outputs, dim=0)

        # 确保输出大小与输入batch_size一致
        if output.size(0) != batch_size:
            output = output[:batch_size]

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
