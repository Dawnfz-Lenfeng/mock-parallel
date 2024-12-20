import torch
from torchsummary import summary

from src.models.yolo import myYOLO

if __name__ == "__main__":
    IMSIZE = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = myYOLO(device=device, input_size=IMSIZE).to(device)

    # 打印模型结构
    summary(model, (3, IMSIZE, IMSIZE))

    # 测试前向传播
    x = torch.randn(1, 3, IMSIZE, IMSIZE).to(device)

    # 测试训练模式
    model.train()
    model.trainable = True
    outputs = model(x)
    print("\nTraining output shapes:")
    print(f"conf_pred: {outputs[0].shape}")
    print(f"cls_pred: {outputs[1].shape}")
    print(f"txtytwth_pred: {outputs[2].shape}")

    # 测试推理模式
    model.eval()
    model.trainable = False
    with torch.no_grad():
        outputs = model(x)
    print("\nInference output shapes:")
    print(f"bboxes: {outputs[0].shape if len(outputs[0]) > 0 else 'empty'}")
    print(f"scores: {outputs[1].shape if len(outputs[1]) > 0 else 'empty'}")
    print(f"labels: {outputs[2].shape if len(outputs[2]) > 0 else 'empty'}")
