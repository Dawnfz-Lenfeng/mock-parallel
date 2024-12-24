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

        # 主模型在第一个GPU上
        self.main_model = self.to(f"cuda:{device_ids[0]}")

        # 在其他GPU上创建参数引用
        self.replicas = []
        for i in range(1, self.num_gpus):
            device = f"cuda:{device_ids[i]}"
            replica = myYOLO(
                device=device,
                input_size=model.input_size,
                num_classes=model.num_classes,
                stride=model.stride,
                conf_thresh=model.conf_thresh,
                nms_thresh=model.nms_thresh,
                trainable=model.trainable,
            ).to(device)

            # 将副本的参数设置为主模型参数的引用
            for param_r, param_m in zip(
                replica.parameters(), self.main_model.parameters()
            ):
                param_r.data = param_m.data
                param_r.requires_grad = True

            self.replicas.append(replica)

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
            device = f"cuda:{self.device_ids[i]}"
            chunk = x[start:end].to(device)
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

        # 主模型处理第一个chunk
        conf_pred, cls_pred, txtytwth_pred = self.main_model(scattered_inputs[0])
        B = scattered_inputs[0].size(0)
        H = W = scattered_inputs[0].size(2) // self.stride
        pred = torch.cat([conf_pred, cls_pred, txtytwth_pred], dim=-1)
        pred = pred.view(B, H, W, -1).permute(0, 3, 1, 2)
        outputs.append(pred)

        # 副本处理剩余chunks
        for i, (chunk, replica) in enumerate(zip(scattered_inputs[1:], self.replicas)):
            conf_pred, cls_pred, txtytwth_pred = replica(chunk)
            B = chunk.size(0)
            H = W = chunk.size(2) // self.stride
            pred = torch.cat([conf_pred, cls_pred, txtytwth_pred], dim=-1)
            pred = pred.view(B, H, W, -1).permute(0, 3, 1, 2)
            outputs.append(pred)

        # 收集并合并输出
        return self.gather_outputs(outputs)

    def reduce_gradients(self):
        """同步所有GPU上的梯度"""
        # 对于每个参数
        for param_idx, param in enumerate(self.main_model.parameters()):
            if param.grad is None:
                continue

            # 收集所有副本的梯度
            grads = [param.grad.data]
            for replica in self.replicas:
                replica_param = list(replica.parameters())[param_idx]
                if replica_param.grad is not None:
                    grads.append(replica_param.grad.data)

            # 计算平均梯度
            grad = torch.stack(grads).mean(dim=0)

            # 更新主模型的梯度
            param.grad.data = grad

            # 更新副本的梯度（虽然不是必需的，但保持一致性）
            for replica in self.replicas:
                replica_param = list(replica.parameters())[param_idx]
                if replica_param.grad is not None:
                    replica_param.grad.data = grad
