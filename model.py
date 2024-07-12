import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

class ChannelAttention(nn.Module):
    def __init__(self, in_channels=64, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class ChannelAttentionLayer(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ChannelAttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y

class RoutingLayer(nn.Module):
    def __init__(self, in_channels, num_experts):
        super(RoutingLayer, self).__init__()
        self.fc = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        x = torch.flatten(x, start_dim=3)
        x = self.fc(x)
        return F.sigmoid(x)

class KernelNonLocalConv(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3, reduction_ratio=2):
        super(KernelNonLocalConv, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.inter_channels = self.in_channels

        self.span_conv = nn.Conv2d(self.in_channels, self.kernel_size**2, kernel_size=3, padding=1)
        self.channel_conv = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1)

        self.down_conv = nn.Conv2d(self.kernel_size**2, self.in_channels, kernel_size=1, padding=0)
        self.down_conv1 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=2, padding=1)
        self.down_conv2 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=2, padding=1)

        self.up_conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1)
        )
        self.up_conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1)
        )
        self.up_conv = nn.Conv2d(self.in_channels, self.kernel_size**2, kernel_size=1, padding=0)

        self.g_conv = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)
        self.W_z_conv = nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1)

        nn.init.constant_(self.W_z_conv.weight, 0)
        nn.init.constant_(self.W_z_conv.bias, 0)

        self.theta_conv = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)
        self.phi_conv = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)

        self.unfold = nn.Unfold(kernel_size=self.kernel_size, dilation=1, padding=1, stride=1)

        self.conv_weight = nn.Conv2d(self.in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(self.in_channels, self.in_channels//2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.in_channels//2, self.in_channels, kernel_size=3, stride=1, padding=1)
        self.weight = nn.Parameter(torch.Tensor(8, self.in_channels, self.in_channels, 1, 1))

        self.avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        self.routing_fn = RoutingLayer(self.in_channels, 8)

    def forward(self, x):
        batch_size, xC, xH, xW = x.shape

        span = self.span_conv(x)
        b, c, h, w = span.shape
        channel_feature = self.channel_conv(x).view(b, self.in_channels, 1, h, w)
        span = span.view(b, self.kernel_size**2, h, w).unsqueeze(1)

        kernel = self.down_conv(span)
        kernel = self.down_conv1(kernel)
        kernel = self.down_conv2(kernel)

        g = self.g_conv(kernel).view(batch_size, self.inter_channels, -1)
        g = g.permute(0, 2, 1)
        theta = self.theta_conv(kernel).view(batch_size, self.inter_channels, -1)
        phi = self.phi_conv(kernel).view(batch_size, self.inter_channels, -1)
        theta = theta.permute(0, 2, 1)
        attention_map = torch.matmul(theta, phi)
        attention_map = attention_map / attention_map.size(-1)
        y = torch.matmul(attention_map, g)
        y = y.permute(0, 2, 1).contiguous().view(batch_size, self.inter_channels, *kernel.size()[2:])
        y = self.W_z_conv(y)
        kernel = y + kernel
        kernel = self.up_conv1(kernel)
        kernel = self.up_conv2(kernel)
        kernel = self.up_conv(kernel).view(b, self.kernel_size**2, h, w).unsqueeze(1)
        kernel = kernel * channel_feature
        kernel = kernel.view([batch_size, xC, self.kernel_size**2, xH, xW])
        kernel = kernel.view(batch_size, xC * self.kernel_size**2, xH, xW)
        kernel = kernel.view([batch_size, xC, -1, xH, xW])

        unfold_feature = self.unfold(x).view([batch_size, xC, -1, xH, xW])
        x = (unfold_feature * kernel).sum(2)

        feat = x
        x = self.conv1(x)
        routing_weights = self.routing_fn(x.permute(0, 2, 3, 1))
        weight_reshaped = self.weight.view(1, 1, 1, 8, -1)
        result = routing_weights.view(batch_size, xH, xW, 8, 1) * weight_reshaped
        result_summed = result.sum(dim=3)
        conv_kernel = result_summed.view(batch_size, xH, xW, self.in_channels, self.in_channels).permute(0, 3, 4, 1, 2)
        ufx = x.view([batch_size, xC, -1, xH, xW])
        output = (ufx * conv_kernel).sum(2)
        output = self.conv2(output)
        
        return output + feat

class KNLNet(nn.Module):
    def __init__(self):
        super(KNLNet, self).__init__()
        self.input_channels = 34
        self.output_channels = 31

        self.inner_channels = 64
        self.initial_conv = nn.Conv2d(self.input_channels, self.inner_channels, kernel_size=3, stride=1, padding=1)

        self.res_blocks = nn.Sequential(*[ResidualBlock() for _ in range(3)])
        self.relu = nn.ReLU()
        self.final_conv = nn.Conv2d(self.inner_channels, self.output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, y):
        input = torch.cat((x, y), 1)
        feature = self.initial_conv(input)
        res_feature = self.res_blocks(feature)
        output = self.final_conv(res_feature) + x
        return output

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.kernel_size = 3
        self.inner_channels = 64
        self.layers = nn.Sequential(
            nn.Conv2d(self.inner_channels, self.inner_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.inner_channels, self.inner_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        res = self.layers(x)
        res += x
        return res
    
