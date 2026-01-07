import torch
import torch.nn as nn

class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA)
    Paper: https://arxiv.org/abs/1910.03151
    """
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        y = self.avg_pool(x)              # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv(y)                  # [B, 1, C]
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y

class ECA_C2f(nn.Module):
    """
    C2f block with Efficient Channel Attention (ECA)
    """
    def __init__(self, c1, c2, n=1, shortcut=False, eca_kernel=3):
        super().__init__()
        self.c = c2 // 2

        self.cv1 = nn.Conv2d(c1, c2, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)

        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut)
            for _ in range(n)
        )

        self.cv2 = nn.Conv2d(c2 + n * self.c, c2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)

        self.act = nn.SiLU(inplace=True)
        self.eca = ECA(c2, k_size=eca_kernel)
  def forward(self, x):
        y = self.act(self.bn1(self.cv1(x)))
        y1, y2 = y.chunk(2, dim=1)

        outs = [y1]
        for block in self.m:
            y2 = block(y2)
            outs.append(y2)

        out = torch.cat(outs, dim=1)
        out = self.act(self.bn2(self.cv2(out)))
        out = self.eca(out)   # 🔥 Attention applied here
        return out
