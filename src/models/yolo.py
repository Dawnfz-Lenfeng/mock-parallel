import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..config import Config
from .blocks import SPP, Conv
from .resnet import ResNet34


class YOLODataset(Dataset):
    """YOLO格式数据集的加载器"""

    def __init__(
        self,
        image_files: list[str],  # 图片文件路径列表
        labels_dir: str,  # 标签文件夹路径
        transform: transforms.Compose | None = None,  # 数据预处理
    ) -> None:
        self.image_files = image_files
        self.labels_dir = labels_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list[list[float]]]:
        """
        返回:
            img: 预处理后的图片张量
            target: [[class_id, x, y, w, h], ...] 格式的标签列表
        """
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        target: list[list[float]] = []
        label_path = os.path.join(
            self.labels_dir, img_path.split("/")[-1].replace(".png", ".txt")
        )
        with open(label_path, "r") as f:
            for line in f:
                class_id, x, y, w, h = map(float, line.split())
                target.append([class_id, x, y, w, h])

        return img, target


class myYOLO(nn.Module):
    def __init__(
        self,
        num_classes=Config.NUM_CLASSES,
        stride=Config.STRIDE,
        conf_thresh=Config.CONF_THRESH,
        trainable=Config.TRAINABLE,
    ):
        super(myYOLO, self).__init__()
        self.num_classes = num_classes  # 类别数
        self.stride = stride  # 网格最大步长
        self.conf_thresh = conf_thresh  # 得分阈值
        self.trainable = trainable  # 训练标识

        # backbone:ResNet34
        """
        Input: [B,3,H,W]
        Output:[B,512,H/32,W/32]
        """
        self.backbone = ResNet34()

        # neck:SPP
        """
        Input:[B,512,H/32,W/32]
        Output:[B,512,H/32,W/32]
        """
        self.neck = nn.Sequential(
            SPP(),  # [B,512,H/32,W/32] to [B,512*4,H/32,W/32]
            Conv(512 * 4, 512, k=1),  # [B,512*4,H/32,W/32] to [B,512,H/32,W/32]
        )

        # detection head
        """
        Input:[B,512,H/32,W/32]
        Output:[B,512,H/32,W/32]
        """
        self.convs = nn.Sequential(
            Conv(512, 216, k=1),
            Conv(216, 512, k=3, p=1),
            Conv(512, 216, k=1),
            Conv(216, 512, k=3, p=1),
        )

        # pred
        """
        Input:[B,512,H/32,W/32]
        Output:[B, 1 + self.num_classes + 4, H/32, W/32]
        """
        self.pred = nn.Conv2d(512, 1 + self.num_classes + 4, 1)

    def create_grid(self, input_size=224):
        """
        用于生成G矩阵，其中每个元素都是特征图上的像素坐标。
        """
        # 输入图像的宽和高
        w, h = input_size, input_size
        # 特征图的宽和高
        ws, hs = w // self.stride, h // self.stride
        # 生成网格的x坐标和y坐标
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        # 将xy两部分的坐标拼起来：[H, W, 2]
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        # [H, W, 2] -> [HW, 2] -> [HW, 2]
        grid_xy = grid_xy.view(-1, 2).to(self.device)

        return grid_xy

    def set_grid(self, input_size=224):
        """
        用于重置G矩阵。
        """
        self.input_size = input_size
        self.grid_cell = self.create_grid(input_size)

    def decode_boxes(self, pred):
        """
        将txtytwth转换为常用的x1y1x2y2形式。
        """
        output = torch.zeros_like(pred)
        # 得到所有bbox 的中心点坐标和宽高
        pred[..., :2] = torch.sigmoid(pred[..., :2]) + self.grid_cell
        pred[..., 2:] = torch.exp(pred[..., 2:])

        # 将所有bbox的中心带你坐标和宽高换算成x1y1x2y2形式
        output[..., :2] = pred[..., :2] * self.stride - pred[..., 2:] * 0.5
        output[..., 2:] = pred[..., :2] * self.stride + pred[..., 2:] * 0.5

        return output

    def nms(self, bboxes, scores):
        """ "Pure Python NMS baseline."""
        x1 = bboxes[:, 0]  # xmin
        y1 = bboxes[:, 1]  # ymin
        x2 = bboxes[:, 2]  # xmax
        y2 = bboxes[:, 3]  # ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # 计算交集的左上角点和右下角点的坐标
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            # 计算交集的宽高
            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
            # 计算交集的面积
            inter = w * h

            # 计算交并比
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # 滤除超过nms阈值的检测框
            inds = np.where(iou <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def postprocess(self, bboxes, scores):
        """
        Input:
            bboxes: [HxW, 4]
            scores: [HxW, num_classes]
        Output:
            bboxes: [N, 4]
            score:  [N,]
            labels: [N,]
        """

        labels = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), labels)]

        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # NMS
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        return bboxes, scores, labels

    @torch.no_grad()
    def inference(self, x):
        """
        用于进行推理
        """
        # backbone主干网络
        x = self.backbone(x)

        # neck网络
        x = self.neck(x)

        # detection head网络
        x = self.convs(x)

        # 预测层
        pred = self.pred(x)

        # 对pred 的size做一些view调整，便于后续的处理
        # [B, C, H/32, W/32] -> [B, H/32, W/32, C] -> [B, (H/32)*(W/32), C]
        pred = pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

        # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测
        # [B, (H/32)*(W/32), 1]
        conf_pred = pred[..., :1]
        # [B, (H/32)*(W/32), num_cls]
        cls_pred = pred[..., 1 : 1 + self.num_classes]
        # [B, (H/32)*(W/32), 4]
        txtytwth_pred = pred[..., 1 + self.num_classes :]

        # 推理默认batch是1，不需要用batch这个维度，用[0]将其取走。
        conf_pred = conf_pred[0]  # [(H/32)*(W/32), 1]
        cls_pred = cls_pred[0]  # [(H/32)*(W/32), NC]
        txtytwth_pred = txtytwth_pred[0]  # [(H/32)*(W/32), 4]

        # 每个边界框的得分
        scores = torch.sigmoid(conf_pred) * torch.softmax(cls_pred, dim=-1)

        # 解算边界框, 并归一化边界框: [(H/32)*(W/32), 4]
        bboxes = self.decode_boxes(txtytwth_pred) / self.input_size
        bboxes = torch.clamp(bboxes, 0.0, 1.0)

        # 将预测放在cpu处理上，以便进行后处理
        scores = scores.to("cpu").numpy()
        bboxes = bboxes.to("cpu").numpy()

        # 后处理
        bboxes, scores, labels = self.postprocess(bboxes, scores)

        return bboxes, scores, labels

    def forward(self, x):
        if not self.trainable:
            return self.inference(x)
        else:
            # backbone主干网络
            x = self.backbone(x)

            # neck网络
            x = self.neck(x)

            # detection head网络
            x = self.convs(x)

            # 预测层
            pred = self.pred(x)

            # 对pred的size做一些view调整，便于后续的处理
            # [B, C, H/32, W/32] -> [B, H/32, W/32, C] -> [B, (H/32)*(W/32), C]
            pred = pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

            # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测
            # [B, (H/32)*(W/32), 1]
            conf_pred = pred[..., :1]
            # [B, (H/32)*(W/32), num_cls]
            cls_pred = pred[..., 1 : 1 + self.num_classes]
            # [B, (H/32)*(W/32), 4]
            txtytwth_pred = pred[..., 1 + self.num_classes :]

            return conf_pred, cls_pred, txtytwth_pred
