import torch
import torch.nn as nn 
import torchvision.models as models 


class Vgg16(nn.Module):
    def __init__(self, pretrained=True):
        super(Vgg16, self).__init__()
        vgg = models.vgg16(pretrained)
        self.conv = vgg.features
        # max -> avg
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Vgg16_BN(nn.Module):
    def __init__(self, pretrained=True):
        super(Vgg16_BN, self).__init__()
        vgg = models.vgg16_bn(pretrained)
        self.conv = vgg.features
        # max -> avg
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 2)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
