import queue
import threading
from collections import deque

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
            with torch.cuda.stream(stream):
                feat = feat.to(device, non_blocking=True)
                feat = module(feat)
                stream.synchronize()
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

        # 使用双端队列管理流水线
        pipeline = deque()
        max_active = min(len(self.stages), len(micro_batches))

        # 填充流水线
        for i in range(max_active):
            pipeline.append(self._pipeline_forward(micro_batches[i]))

        # 处理剩余的micro-batches
        for i in range(max_active, len(micro_batches)):
            outputs.append(pipeline.popleft())
            pipeline.append(self._pipeline_forward(micro_batches[i]))

        # 清空流水线
        while pipeline:
            outputs.append(pipeline.popleft())

        # 合并输出并处理
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


class ThreadPipeline(myYOLO):
    """基于多线程的流水线并行实现"""

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

        # 为每个阶段创建队列
        self.queues = {
            "input": queue.Queue(),
            "backbone": queue.Queue(),
            "neck": queue.Queue(),
            "convs": queue.Queue(),
            "pred": queue.Queue(),
            "output": queue.Queue(),
        }

        # 分配各阶段到不同GPU
        self._distribute_stages()

        # 创建工作线程
        self.threads = []
        self._create_workers()

    def _distribute_stages(self):
        """将模型的不同阶段分配到不同GPU"""
        self.stages = {
            "backbone": (self.backbone, f"cuda:{self.device_ids[0]}"),
            "neck": (self.neck, f"cuda:{self.device_ids[1 % self.num_gpus]}"),
            "convs": (self.convs, f"cuda:{self.device_ids[2 % self.num_gpus]}"),
            "pred": (self.pred, f"cuda:{self.device_ids[3 % self.num_gpus]}"),
        }

        # 将每个阶段移到对应的GPU
        for name, (module, device) in self.stages.items():
            self.stages[name] = (module.to(device), device)

    def _stage_worker(self, stage_name: str):
        """每个阶段的工作线程"""
        module, device = self.stages[stage_name]
        input_queue = self.queues[stage_name]
        output_queue = (
            self.queues["pred"]
            if stage_name == "convs"
            else self.queues[
                {"backbone": "neck", "neck": "convs", "pred": "output"}[stage_name]
            ]
        )

        while True:
            try:
                data = input_queue.get(timeout=1)
                if data is None:
                    output_queue.put(None)
                    break

                # 处理数据
                data = data.to(device, non_blocking=True)
                with torch.cuda.stream(torch.cuda.Stream(device=device)):
                    output = module(data)
                    torch.cuda.current_stream(device).synchronize()

                output_queue.put(output)

            except queue.Empty:
                continue

    def _create_workers(self):
        """创建所有工作线程"""
        for stage_name in ["backbone", "neck", "convs", "pred"]:
            thread = threading.Thread(target=self._stage_worker, args=(stage_name,))
            thread.daemon = True
            self.threads.append(thread)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.trainable:
            return self.inference(x)

        # 启动所有线程
        for thread in self.threads:
            if not thread.is_alive():
                thread.start()

        batch_size = x.size(0)
        self.chunks = min(self.chunks, batch_size)
        while batch_size % self.chunks != 0:
            self.chunks -= 1

        micro_batches = x.chunk(self.chunks)
        outputs = []

        # 将micro-batches放入输入队列
        for mb in micro_batches:
            self.queues["backbone"].put(mb)

        # 收集输出
        for _ in range(len(micro_batches)):
            output = self.queues["output"].get()
            outputs.append(output)

        # 合并输出并处理
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

    def __del__(self):
        """清理资源"""
        # 发送结束信号
        for queue in self.queues.values():
            queue.put(None)

        # 等待所有线程结束
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1)
