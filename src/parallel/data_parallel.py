import torch
import torch.distributed as dist
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


class MPDataParallel(myYOLO):
    """使用手动复制模型和distributed all_reduce的数据并行实现"""

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

        # 初始化分布式环境
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="tcp://127.0.0.1:23456",
                world_size=1,  # 单进程多GPU
                rank=0,
            )

        # 创建模型副本
        self.models = nn.ModuleList()
        for gpu_id in device_ids:
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
            model_copy.load_state_dict(model.state_dict())
            self.models.append(model_copy)

    def scatter_inputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        """将输入数据分散到不同GPU"""
        batch_size = x.size(0)
        chunks = []
        chunk_sizes = [batch_size // self.num_gpus] * self.num_gpus
        remainder = batch_size % self.num_gpus
        for i in range(remainder):
            chunk_sizes[i] += 1

        start = 0
        for i, size in enumerate(chunk_sizes):
            end = start + size
            device = f"cuda:{self.device_ids[i]}"
            chunk = x[start:end].to(device, non_blocking=True)
            chunks.append(chunk)
            start = end

        return chunks

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
            with torch.cuda.device(f"cuda:{self.device_ids[i]}"):
                conf_pred, cls_pred, txtytwth_pred = self.models[i](chunk)
                B = chunk.size(0)
                H = W = chunk.size(2) // self.stride
                pred = torch.cat([conf_pred, cls_pred, txtytwth_pred], dim=-1)
                pred = pred.view(B, H, W, -1).permute(0, 3, 1, 2)
                outputs.append(pred)

        # 收集并合并输出
        return self.gather_outputs(outputs)

    def gather_outputs(
        self, outputs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """收集并合并所有GPU的输出"""
        main_device = f"cuda:{self.device_ids[0]}"
        output = torch.cat([out.to(main_device) for out in outputs], dim=0)

        pred = output.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        conf_pred = pred[..., :1]
        cls_pred = pred[..., 1 : 1 + self.num_classes]
        txtytwth_pred = pred[..., 1 + self.num_classes :]

        return conf_pred, cls_pred, txtytwth_pred

    def reduce_gradients(self):
        """使用all_reduce同步梯度"""
        for param_idx, param in enumerate(self.models[0].parameters()):
            if param.grad is None:
                continue

            # 收集所有GPU上的梯度
            grads = torch.stack(
                [
                    list(model.parameters())[param_idx].grad.to(self.device)
                    for model in self.models
                ]
            )

            # 使用all_reduce计算平均梯度
            dist.all_reduce(grads)
            avg_grad = grads / self.num_gpus

            # 将平均梯度分发到所有GPU
            for i, model in enumerate(self.models):
                param = list(model.parameters())[param_idx]
                param.grad.data.copy_(avg_grad[i].to(param.device))

    def __del__(self):
        """清理分布式环境"""
        if dist.is_initialized():
            dist.destroy_process_group()


class CustomDataParallel(myYOLO):
    """使用手动梯度同步的数据并行实现"""

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

        # 创建模型副本
        self.models = nn.ModuleList()
        for i, gpu_id in enumerate(device_ids):
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
            model_copy.load_state_dict(model.state_dict())
            self.models.append(model_copy)

    def scatter_inputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        """将输入数据分散到不同GPU"""
        batch_size = x.size(0)
        chunks = []
        chunk_sizes = [batch_size // self.num_gpus] * self.num_gpus
        remainder = batch_size % self.num_gpus
        for i in range(remainder):
            chunk_sizes[i] += 1

        start = 0
        for i, size in enumerate(chunk_sizes):
            end = start + size
            device = f"cuda:{self.device_ids[i]}"
            chunk = x[start:end].to(device, non_blocking=True)
            chunks.append(chunk)
            start = end

        return chunks

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
            with torch.cuda.device(f"cuda:{self.device_ids[i]}"):
                conf_pred, cls_pred, txtytwth_pred = self.models[i](chunk)
                B = chunk.size(0)
                H = W = chunk.size(2) // self.stride
                pred = torch.cat([conf_pred, cls_pred, txtytwth_pred], dim=-1)
                pred = pred.view(B, H, W, -1).permute(0, 3, 1, 2)
                outputs.append(pred)

        # 收集并合并输出
        return self.gather_outputs(outputs)

    def gather_outputs(
        self, outputs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """收集并合并所有GPU的输出"""
        main_device = f"cuda:{self.device_ids[0]}"
        output = torch.cat([out.to(main_device) for out in outputs], dim=0)

        pred = output.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        conf_pred = pred[..., :1]
        cls_pred = pred[..., 1 : 1 + self.num_classes]
        txtytwth_pred = pred[..., 1 + self.num_classes :]

        return conf_pred, cls_pred, txtytwth_pred

    def reduce_gradients(self):
        """手动同步梯度"""
        for param_idx, param in enumerate(self.models[0].parameters()):
            if param.grad is None:
                continue

            # 收集所有GPU上的梯度
            grads = []
            for model in self.models:
                param = list(model.parameters())[param_idx]
                if param.grad is not None:
                    grads.append(param.grad)

            if not grads:
                continue

            # 在第一个GPU上计算平均梯度
            with torch.cuda.device(self.device):
                avg_grad = torch.stack([grad.to(self.device) for grad in grads]).mean(
                    dim=0
                )

            # 将平均梯度分发到所有GPU
            for model in self.models:
                param = list(model.parameters())[param_idx]
                if param.grad is not None:
                    param.grad.data.copy_(avg_grad.to(param.device))
