import numpy as np
import torch


def detection_collate(
    batch: list[tuple[torch.Tensor, list[list[float]]]]
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    将不同长度的目标检测数据整理成batch
    Args:
        batch: [(img, target), ...] 格式的数据列表
    Returns:
        imgs: batch化的图片张量
        targets: batch化的标签列表
    """
    targets: list[torch.Tensor] = []
    imgs: list[torch.Tensor] = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def gt_creator(input_size: int, stride: int, label_lists: list[list[float]]):
    """
    将YOLO格式的标签转换为训练所需格式
    Args:
        input_size: 输入图片大小
        stride: 特征图的步长
        label_lists: [[class_id, x, y, w, h], ...] 格式的标签列表
    Returns:
        处理后的标签张量
    """
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
