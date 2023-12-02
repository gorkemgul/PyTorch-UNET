import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class ConvBnReluBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act = False):
        super().__init__()
        self.convbnrelu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = not bn_act),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = not bn_act),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.convbnrelu(x)

class UNET(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1):
        super().__init__()
        self.features_down = [64, 128, 256, 512]
        self.features_up = [1024, 512, 256, 128]
        self.up = nn.ModuleList()
        self.down = nn.ModuleList()
        self.maxpool = nn.MaxPool2d(2, 2)

        for feature in self.features_down:
            self.down.append(ConvBnReluBlock(in_channels, out_channels = feature))
            in_channels = feature

        for feature in self.features_up:
            self.up.append(nn.ConvTranspose2d(feature, feature // 2, kernel_size = 2, stride = 2))
            self.up.append(ConvBnReluBlock(feature, feature // 2))

        self.bottleneck = ConvBnReluBlock(in_channels = 512, out_channels = 1024)
        self.last_conv = nn.Conv2d(in_channels = 64, out_channels = out_channels, kernel_size = 1)

    def forward(self, x):
        skip_cons = []

        for _down in self.down:
            x = _down(x)
            skip_cons.append(x)
            x = self.maxpool(x)

        x = self.bottleneck(x)
        skip_cons = skip_cons[::-1]

        for idx in range(0, len(self.up), 2):
            x = self.up[idx](x)
            skip_con = skip_cons[idx // 2]
            concat_skip_con = torch.cat((skip_con, x), dim = 1)
            x = self.up[idx + 1](concat_skip_con)

        return self.last_conv(x)