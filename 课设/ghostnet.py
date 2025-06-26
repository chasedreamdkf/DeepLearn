import torch
import torch.nn as nn
import torch.nn.functional as F

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        init_channels = int(oup / ratio)
        new_channels = oup - init_channels
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :out.shape[1], :, :]

class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se=False):
        super(GhostBottleneck, self).__init__()
        self.stride = stride
        self.use_se = use_se
        self.ghost1 = GhostModule(inp, hidden_dim, relu=True)
        if stride > 1:
            self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size//2, groups=hidden_dim, bias=False)
            self.dwbn = nn.BatchNorm2d(hidden_dim)
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, hidden_dim // 4, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 4, hidden_dim, 1, bias=True),
                nn.Sigmoid()
            )
        self.ghost2 = GhostModule(hidden_dim, oup, relu=False)
        if inp == oup and stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size, stride, kernel_size//2, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.ghost1(x)
        if self.stride > 1:
            out = self.dwconv(out)
            out = self.dwbn(out)
        if self.use_se:
            se_weight = self.se(out)
            out = out * se_weight
        out = self.ghost2(out)
        out += self.shortcut(x)
        return out

class GhostNet(nn.Module):
    def __init__(self, num_classes=1000, width=1.0):
        super(GhostNet, self).__init__()
        cfgs = [  # k, t, c, SE, s
            # stage1
            [3,  16,  16, 0, 1],
            # stage2
            [3,  48,  24, 0, 2],
            [3,  72,  24, 0, 1],
            # stage3
            [5,  72,  40, 1, 2],
            [5, 120,  40, 1, 1],
            # stage4
            [3, 240,  80, 0, 2],
            [3, 200,  80, 0, 1],
            [3, 184,  80, 0, 1],
            [3, 184,  80, 0, 1],
            [3, 480, 112, 1, 1],
            [3, 672, 112, 1, 1],
            # stage5
            [5, 672, 160, 1, 2],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 1, 1],
            [5, 960, 160, 0, 1],
        ]
        output_channel = int(16 * width)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel
        stages = []
        block = GhostBottleneck
        for k, exp_size, c, se, s in cfgs:
            output_channel = int(c * width)
            hidden_channel = int(exp_size * width)
            stages.append(block(input_channel, hidden_channel, output_channel, k, s, se==1))
            input_channel = output_channel
        self.blocks = nn.Sequential(*stages)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(input_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# 用于去雾任务的特征提取，可以只用GhostNet的前几层作为编码器，后续解码器可自定义。
class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class DehazeNet(nn.Module):
    def __init__(self, base_channel=16):
        super(DehazeNet, self).__init__()
        self.encoder = GhostNet(width=1.0)
        self.enc_layers = [3, 6, 9, 12, 14]
        # 解码器，增大通道数以增强模型表达能力
        self.up4 = UpBlock(160, 160, 256)  # enc5, enc4 -> 256
        self.up3 = UpBlock(256, 112, 128)  # 上一步, enc3 -> 128
        self.up2 = UpBlock(128, 80, 64)    # 上一步, enc2 -> 64
        self.up1 = UpBlock(64, 40, 32)     # 上一步, enc1 -> 32
        self.up0 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(16, 3, kernel_size=1)

    def forward(self, x):
        # 编码器前向，保存跳跃连接特征
        feats = []
        out = self.encoder.conv_stem(x)
        out = self.encoder.bn1(out)
        out = self.encoder.act1(out)
        for i, block in enumerate(self.encoder.blocks):
            out = block(out)
            if i in self.enc_layers:
                feats.append(out)
        # 打印特征层shape，调试用
        # for idx, f in enumerate(feats):
        #     print(f"feats[{idx}] shape:", f.shape)
        # feats: [enc1, enc2, enc3, enc4, enc5]
        enc1, enc2, enc3, enc4, enc5 = feats
        d4 = self.up4(enc5, enc4)
        d3 = self.up3(d4, enc3)
        d2 = self.up2(d3, enc2)
        d1 = self.up1(d2, enc1)
        d0 = self.up0(d1)
        out = self.final_conv(d0)
        out = torch.sigmoid(out)
        # 保证输出分辨率与输入一致
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out 