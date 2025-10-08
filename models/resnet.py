import torch.nn as nn
import torchvision.models as models


class ResNetMNIST(nn.Module):
    def __init__(self, num_classes=10, version=18):
        super(ResNetMNIST, self).__init__()

        if version == 18:
            self.resnet = models.resnet18(pretrained=False)
        elif version == 34:
            self.resnet = models.resnet34(pretrained=False)
        else:
            raise ValueError(f"Unsupported ResNet version: {version}")

        # 修改第一层卷积，适配单通道输入
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 修改最后的全连接层
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)