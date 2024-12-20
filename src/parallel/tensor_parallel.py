import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from ..models import myYOLO


class TensorParallel:
    """
    张量并行接口
    """

    def __init__(self, model: myYOLO, device_ids: list[int]):
        self.model = model
        self.device_ids = device_ids
        self.num_gpus = len(device_ids)

        # 确保模型在第一个GPU上
        self.model = self.model.to(f'cuda:{device_ids[0]}')

        # 记录需要进行张量并行的层
        self.parallel_layers = self._get_parallel_layers()

    def _get_parallel_layers(self) -> Dict[str, nn.Module]:
        """获取需要并行化的层"""
        return {
            'backbone': self.model.backbone,
            'neck': self.model.neck,
            'convs': self.model.convs,
            'pred': self.model.pred
        }

    def parallelize(self) -> nn.Module:
        """实现张量并行"""
        # 为每个GPU分配部分模型
        for gpu_id in self.device_ids:
            device = f'cuda:{gpu_id}'
            # 复制模型到每个GPU
            if gpu_id != self.device_ids[0]:
                self.model = self.model.to(device)
        
        # 修改前向传播方法
        def forward(x: torch.Tensor) -> torch.Tensor:
            if self.model.trainable:
                return self._forward_train(x)
            else:
                return self._forward_inference(x)
                
        self.model.forward = forward
        return self.model

    def _forward_train(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """训练模式的前向传播"""
        outputs = []
        chunk_size = x.size(0) // self.num_gpus
        
        # 将输入分到不同GPU
        for i, gpu_id in enumerate(self.device_ids):
            if i == len(self.device_ids) - 1:
                # 最后一个GPU处理剩余的所有数据
                chunk = x[i*chunk_size:].to(f'cuda:{gpu_id}')
            else:
                chunk = x[i*chunk_size:(i+1)*chunk_size].to(f'cuda:{gpu_id}')
            
            # 在每个GPU上进行计算
            out = self.model(chunk)
            outputs.append(out)
            
        # 在第一个GPU上合并结果
        output = torch.cat(outputs, dim=0).to(f'cuda:{self.device_ids[0]}')
        
        # 从输出中分离出三个预测
        pred = output.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        conf_pred = pred[..., :1]
        cls_pred = pred[..., 1:1+self.model.num_classes]
        txtytwth_pred = pred[..., 1+self.model.num_classes:]
        
        return conf_pred, cls_pred, txtytwth_pred

    def _forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        """推理模式的前向传播"""
        outputs = []
        chunk_size = x.size(0) // self.num_gpus
        
        for i, gpu_id in enumerate(self.device_ids):
            if i == len(self.device_ids) - 1:
                chunk = x[i*chunk_size:].to(f'cuda:{gpu_id}')
            else:
                chunk = x[i*chunk_size:(i+1)*chunk_size].to(f'cuda:{gpu_id}')
            
            out = self.model(chunk)
            outputs.append(out)
            
        return torch.cat(outputs, dim=0).to(f'cuda:{self.device_ids[0]}')
