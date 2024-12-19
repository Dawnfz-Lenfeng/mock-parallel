import numpy as np
import torch
import torch.nn as nn
from config import Config
from models.blocks import SPP, Conv
from models.resnet import ResNet34
from torchsummary import summary


class MSEWithLogitsLoss(nn.Module):
    def __init__(
        self,
    ):
        super(MSEWithLogitsLoss, self).__init__()

    def forward(self, logits, target):
        inputs = torch.clamp(torch.sigmoid(logits), min=1e-4, max=1.0 - 1e-4)

        pos_id = (target == 1.0).float()
        neg_id = (target == 0.0).float()
        pos_loss = pos_id * (inputs - target) ** 2
        neg_loss = neg_id * (inputs) ** 2
        loss = 5.0 * pos_loss + 1.0 * neg_loss

        return loss


def generate_dxdywh(gt_label, w, h, s):
    x, y, bw, bh = gt_label[1:]  # 提取x,y,w,h（这四个数都经过了归一化）
    # 计算边界框的中心点
    c_x = x * w  # 乘以图像宽度得到真实坐标
    c_y = y * h  # 乘以图像高度得到真实坐标
    box_w = bw * w  # 乘以图像宽度得到框的真实宽度
    box_h = bh * h  # 乘以图像高度得到框的真实高度

    if box_w < 1e-4 or box_h < 1e-4:
        # print('Not a valid data !!!')
        return False

    # 计算中心点所在的网格坐标
    c_x_s = c_x / s
    c_y_s = c_y / s
    grid_x = int(c_x_s)
    grid_y = int(c_y_s)
    # 计算中心点偏移量和宽高的标签
    tx = c_x_s - grid_x
    ty = c_y_s - grid_y
    tw = np.log(box_w)
    th = np.log(box_h)
    # 计算边界框位置参数的损失权重
    weight = 2.0 - (box_w / w) * (box_h / h)

    return grid_x, grid_y, tx, ty, tw, th, weight


def gt_creator(input_size, stride, label_lists=[]):
    # 必要的参数
    batch_size = len(label_lists)
    w = input_size
    h = input_size
    ws = w // stride
    hs = h // stride
    s = stride
    gt_tensor = np.zeros([batch_size, hs, ws, 1 + 1 + 4 + 1])

    # 制作训练标签
    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]:
            gt_class = int(gt_label[0])
            result = generate_dxdywh(gt_label, w, h, s)
            if result:
                grid_x, grid_y, tx, ty, tw, th, weight = result

                if grid_x < gt_tensor.shape[2] and grid_y < gt_tensor.shape[1]:
                    gt_tensor[batch_index, grid_y, grid_x, 0] = 1.0
                    gt_tensor[batch_index, grid_y, grid_x, 1] = gt_class
                    gt_tensor[batch_index, grid_y, grid_x, 2:6] = np.array(
                        [tx, ty, tw, th]
                    )
                    gt_tensor[batch_index, grid_y, grid_x, 6] = weight

    gt_tensor = gt_tensor.reshape(batch_size, -1, 1 + 1 + 4 + 1)

    return torch.from_numpy(gt_tensor).float()


def compute_loss(pred_conf, pred_cls, pred_txtytwth, targets):
    batch_size = pred_conf.size(0)
    # 损失函数
    conf_loss_function = MSEWithLogitsLoss()
    cls_loss_function = nn.CrossEntropyLoss(reduction="none")
    txty_loss_function = nn.BCEWithLogitsLoss(reduction="none")
    twth_loss_function = nn.MSELoss(reduction="none")

    # 预测
    pred_conf = pred_conf[:, :, 0]  # [B, (H/32)*(W/32),]
    pred_cls = pred_cls.permute(0, 2, 1)  # [B, C, (H/32)*(W/32)]
    pred_txty = pred_txtytwth[:, :, :2]  # [B, (H/32)*(W/32), 2]
    pred_twth = pred_txtytwth[:, :, 2:]  # [B, (H/32)*(W/32), 2]

    # 标签
    gt_obj = targets[:, :, 0]  # [B, (H/32)*(W/32),]
    gt_cls = targets[:, :, 1].long()  # [B, (H/32)*(W/32),]
    gt_txty = targets[:, :, 2:4]  # [B, (H/32)*(W/32), 2]
    gt_twth = targets[:, :, 4:6]  # [B, (H/32)*(W/32), 2]
    gt_box_scale_weight = targets[:, :, 6]  # [B, (H/32)*(W/32),]

    batch_size = pred_conf.size(0)

    # 置信度损失
    conf_loss = conf_loss_function(pred_conf, gt_obj)
    conf_loss = conf_loss.sum() / batch_size

    # 类别损失
    cls_loss = cls_loss_function(pred_cls, gt_cls) * gt_obj
    cls_loss = cls_loss.sum() / batch_size

    # 边界框txty的损失
    txty_loss = (
        txty_loss_function(pred_txty, gt_txty).sum(-1) * gt_obj * gt_box_scale_weight
    )
    txty_loss = txty_loss.sum() / batch_size

    # 边界框twth的损失
    twth_loss = (
        twth_loss_function(pred_twth, gt_twth).sum(-1) * gt_obj * gt_box_scale_weight
    )
    twth_loss = twth_loss.sum() / batch_size
    bbox_loss = txty_loss + twth_loss

    # 总的损失
    total_loss = conf_loss + cls_loss + bbox_loss

    return conf_loss, cls_loss, bbox_loss, total_loss


class myYOLO(nn.Module):
    def __init__(
        self, num_classes=Config.NUM_CLASSES, stride=Config.STRIDE, conf_thresh=Config.CONF_THRESH, trainable=Config.TRAINABLE
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


if __name__ == "__main__":
    IMSIZE = 224
    my_YOLO = myYOLO().cuda()

    summary(my_YOLO, [(3, IMSIZE, IMSIZE)])
