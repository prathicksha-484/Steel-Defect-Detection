import torch
import torch.nn as nn
import timm

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        return x * self.ca(x) * self.sa(x)

class AttentiveMobileNetV3(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('mobilenetv3_large_100', pretrained=pretrained, in_chans=1, features_only=True)
        channels = self.backbone.feature_info[-1]['num_chs']
        self.cbam = CBAM(channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels, 128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)[-1]
        feats = self.cbam(feats)
        return self.classifier(self.pool(feats).flatten(1))

def build_model(name="resnet18", num_classes=6):
    if name == "attentive_mobilenetv3":
        return AttentiveMobileNetV3(num_classes=num_classes)
    return timm.create_model(name, pretrained=True, num_classes=num_classes, in_chans=1)