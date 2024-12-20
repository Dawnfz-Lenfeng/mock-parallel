import torch
import torch.nn as nn
from typing import List, Tuple

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

        # 确保模型在第一个GPU上
        self.model = self.model.to(f'cuda:{device_ids[0]}')

        # 计算每个阶段的设备
        self.stage_devices = self._assign_devices()

        # 将模型分成几个阶段并分配到不同GPU
        self.stages = self._create_stages()

    def _assign_devices(self) -> List[int]:
        """为每个阶段分配GPU"""
        num_stages = 4  # backbone, neck, convs, pred
        return [self.device_ids[i % self.num_gpus] for i in range(num_stages)]

    def _create_stages(self) -> List[nn.Module]:
        """创建并分配各个阶段"""
        stages = []
        # backbone
        backbone = nn.Sequential(self.model.backbone).to(f'cuda:{self.stage_devices[0]}')
        stages.append(backbone)
        
        # neck
        neck = nn.Sequential(self.model.neck).to(f'cuda:{self.stage_devices[1]}')
        stages.append(neck)
        
        # convs
        convs = nn.Sequential(self.model.convs).to(f'cuda:{self.stage_devices[2]}')
        stages.append(convs)
        
        # pred
        pred = nn.Sequential(self.model.pred).to(f'cuda:{self.stage_devices[3]}')
        stages.append(pred)
        
        return stages

    def parallelize(self) -> nn.Module:
        """实现流水线并行"""
        def forward(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if self.model.trainable:
                return self._forward_train(x)
            else:
                return self._forward_inference(x)
                
        self.model.forward = forward
        return self.model

    def _forward_train(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """训练模式的前向传播"""
        # 将输入分成chunks个micro-batch
        micro_batches = x.chunk(self.chunks)
        outputs = []
        
        for mb in micro_batches:
            current_input = mb
            # 通过每个阶段
            for stage_idx, stage in enumerate(self.stages):
                current_device = self.stage_devices[stage_idx]
                current_input = current_input.to(f'cuda:{current_device}')
                current_input = stage(current_input)
            
            # 确保输出在最后一个设备上
            outputs.append(current_input)
        
        # 合并所有micro-batch的输出
        output = torch.cat(outputs, dim=0)
        
        # 从最后一层的输出中分离出三个预测
        pred = output.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        conf_pred = pred[..., :1]
        cls_pred = pred[..., 1:1+self.model.num_classes]
        txtytwth_pred = pred[..., 1+self.model.num_classes:]
        
        return conf_pred, cls_pred, txtytwth_pred

    def _forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        """推理模式的前向传播"""
        current_input = x
        for stage_idx, stage in enumerate(self.stages):
            current_device = self.stage_devices[stage_idx]
            current_input = current_input.to(f'cuda:{current_device}')
            current_input = stage(current_input)
        return current_input
