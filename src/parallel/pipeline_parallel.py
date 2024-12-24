import torch

from ..yolo import myYOLO


class StreamPipeline(myYOLO):
    """基于CUDA Stream的流水线并行实现"""

    def __init__(self, model: myYOLO, device_ids: list[int], chunks: int):
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
        self.chunks = chunks
        self.num_gpus = len(device_ids)
        self.load_state_dict(model.state_dict())

        # 创建CUDA流
        self.streams = {
            device: torch.cuda.Stream(device=device)
            for device in [f"cuda:{i}" for i in device_ids]
        }

        # 分配各阶段到不同GPU
        self._distribute_stages()

    def _distribute_stages(self):
        """将模型的不同阶段分配到不同GPU"""
        self.stages = [
            (self.backbone, f"cuda:{self.device_ids[0]}"),
            (self.neck, f"cuda:{self.device_ids[1 % self.num_gpus]}"),
            (self.convs, f"cuda:{self.device_ids[2 % self.num_gpus]}"),
            (self.pred, f"cuda:{self.device_ids[3 % self.num_gpus]}"),
        ]

        # 将每个阶段移到对应的GPU
        for i, (module, device) in enumerate(self.stages):
            self.stages[i] = (module.to(device), device)

    def _pipeline_forward(self, x: torch.Tensor) -> torch.Tensor:
        """单个micro-batch的流水线前向传播"""
        feat = x
        for module, device in self.stages:
            stream = self.streams[device]
            # Wait for current stream's previous work to complete
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                feat = feat.to(device, non_blocking=True)
                feat = module(feat)
            # Make the default stream wait for this stream
            torch.cuda.current_stream(device).wait_stream(stream)
        return feat

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.trainable:
            return self.inference(x)

        batch_size = x.size(0)
        self.chunks = min(self.chunks, batch_size)
        while batch_size % self.chunks != 0:
            self.chunks -= 1

        micro_batches = x.chunk(self.chunks)
        outputs = []

        # 初始化每个时间片的状态
        # 每个元素是 (feature, current_stage_index)
        current_state = [(micro_batches[0], 0)]  # 从第一个batch开始
        next_batch_idx = 1

        while len(outputs) < len(micro_batches):
            next_state = []

            # 在当前时间片内，处理所有活跃的batch
            for feat, stage_idx in current_state:
                module, device = self.stages[stage_idx]
                stream = self.streams[device]

                with torch.cuda.stream(stream):
                    feat = feat.to(device, non_blocking=True)
                    feat = module(feat)

                # 记录这个batch的下一个状态
                if stage_idx == len(self.stages) - 1:
                    outputs.append(feat)
                else:
                    next_state.append((feat, stage_idx + 1))

            # 同步所有stream，确保当前时间片的所有操作都完成
            for stream in self.streams.values():
                stream.synchronize()

            # 如果还有未处理的batch且pipeline未满，加入新的batch
            if next_batch_idx < len(micro_batches) and len(next_state) < len(
                self.stages
            ):
                next_state.append((micro_batches[next_batch_idx], 0))
                next_batch_idx += 1

            current_state = next_state

        output = torch.cat(outputs, dim=0)
        pred = output.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

        # 分离预测结果
        conf_pred = pred[..., :1]
        cls_pred = pred[..., 1 : 1 + self.num_classes]
        txtytwth_pred = pred[..., 1 + self.num_classes :]

        # 移到主设备
        main_device = f"cuda:{self.device_ids[0]}"
        conf_pred = conf_pred.to(main_device)
        cls_pred = cls_pred.to(main_device)
        txtytwth_pred = txtytwth_pred.to(main_device)

        return conf_pred, cls_pred, txtytwth_pred
