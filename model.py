import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as int
# from torchstat import stat
from torch.nn.parameter import Parameter
import torch.nn.functional as F
# #HYPER PARAMS(Pre-Defined) #
# lr = 0.00001
import functools

class ChannelAttention(nn.Module):
    def __init__(self, in_planes=64, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 8, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 8, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y





class _routing(nn.Module):

    def __init__(self, in_channels, num_experts):
        super(_routing, self).__init__()

        self.fc = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        x = torch.flatten(x)
        x = self.fc(x)
        return F.sigmoid(x)


class KNLConv(nn.Module):
    def __init__(self,in_channel=64,kernel_size=3,red_ratio=2):
        super(KNLConv, self).__init__()
        self.in_channels=in_channel
        self.outChannel = in_channel
        self.kernel_size = kernel_size
        self.red_ratio=red_ratio
        self.inter_channels=self.in_channels

        self.span = nn.Conv2d(self.in_channels, self.kernel_size**2, kernel_size=3,padding=1)

        self.chConv = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1,padding=1)


        nlchan = self.in_channels * (self.kernel_size ** 2)
        # non-local
        self.dimension=2

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.conv_num = 8

        self.down = conv_nd(in_channels = (self.kernel_size ** 2),
                             out_channels=self.in_channels,
                             stride=1, kernel_size=1, padding=0)
        self.down1 = conv_nd(in_channels=self.in_channels,
                             out_channels=self.in_channels,
                             stride=2, kernel_size=3, padding=1)

        self.down2 = conv_nd(in_channels=self.in_channels,
                             out_channels=self.in_channels,
                             stride=2, kernel_size=3, padding=1)

        self.up1=nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=self.in_channels,
                          out_channels=self.in_channels,
                          kernel_size=3, stride=1, padding=1))
        self.up2=nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=self.in_channels,
                          out_channels=self.in_channels,
                          kernel_size=3, stride=1, padding=1))
        self.up=nn.Conv2d(in_channels=self.in_channels,
                  out_channels=(self.kernel_size ** 2),
                  kernel_size=1, stride=1, padding=0)

        self.g = conv_nd(in_channels=self.in_channels ,
                         out_channels=self.inter_channels, kernel_size=1)

        self.W_z = conv_nd(in_channels=self.inter_channels,
                           out_channels=self.in_channels, kernel_size=1)

        nn.init.constant_(self.W_z.weight, 0)
        nn.init.constant_(self.W_z.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels, kernel_size=1)

        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels, kernel_size=1)

        # self.relu = nn.ReLU(inplace=True)
        self.unfold1 = nn.Unfold(kernel_size=self.kernel_size, dilation=1, padding=1, stride=1)



        self.conv_weight =  nn.Conv2d(in_channels=self.in_channels, out_channels=self.conv_num, kernel_size=3, stride=1, padding=1)

        self.weight = Parameter(torch.Tensor(self.conv_num, self.in_channels, self.in_channels, 1, 1))

        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        self._routing_fn = _routing(self.in_channels, self.conv_num)


    def forward(self, feature):

        BS, xC, xH, xW = feature.size()

        x1=self.span(feature)
        b, c, h, w = x1.shape
        ch_fea=self.chConv(feature).view(b, self.in_channels, 1, h, w)
        x1 = x1.view(b, self.kernel_size ** 2, h, w).unsqueeze(1)

        ker1 = x1.reshape(BS, (self.kernel_size ** 2), xH, xW)

        # print(ker1.shape)
        fea_ker = self.down(ker1)
        fea_ker=self.down1(fea_ker )
        fea_ker=self.down2(fea_ker)
        # print(fea_ker.shape)
        g_x = self.g(fea_ker).view(BS, self.inter_channels, -1)
        # print(g_x.shape)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(fea_ker).view(BS, self.inter_channels, -1)
        phi_x = self.phi(fea_ker).view(BS, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        f = torch.matmul(theta_x, phi_x)
        N1 = f.size(-1)  # number of position in x
        f = f / N1
        y = torch.matmul(f, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(BS, self.inter_channels, * fea_ker.size()[2:])
        y = self.W_z(y)
        kernel = y + fea_ker
        kernel=self.up1(kernel)
        kernel=self.up2(kernel)
        kernel = self.up(kernel).view(b, self.kernel_size ** 2, h, w).unsqueeze(1)
        kernel = kernel * ch_fea
        kernel = kernel.reshape([BS, xC, (self.kernel_size ** 2), xH, xW])
        kernel = kernel.reshape(BS, xC * (self.kernel_size ** 2), xH, xW)
        kernel=kernel.reshape([BS,xC, -1, xH, xW])


        unfold_feature = self.unfold1(feature).reshape([BS,xC, -1, xH, xW])
        x = (unfold_feature * kernel).sum(2)

        x1 = x.unsqueeze(0)
        pooled_inputs = self._avg_pooling(x1)
        routing_weights = self._routing_fn(pooled_inputs)

        kernels = torch.sum(routing_weights[:, None, None, None, None] * self.weight, 0)

        x = F.conv2d(x, kernels)


        return x


class KNLNet(nn.Module):
    def __init__(self):
        super(KNLNet, self).__init__()
        self.inC = 31 + 3
        self.outC = 31

        self.kernel_size = 3
        self.block_number = 3
        self.innerC = 64
        self.inConv = nn.Conv2d(in_channels=self.inC, out_channels=self.innerC, kernel_size=3, stride=1, padding=1, bias=True)

        layers = []
        for i in range(self.block_number):
            layers.append(ResBlock())
        self.adConvList=nn.Sequential(*layers)
        self.relu=nn.ReLU()
        self.outConv = nn.Conv2d(in_channels=self.innerC, out_channels=self.outC, kernel_size=3, stride=1, padding=1,
                                bias=True)

    def forward(self, x,y):
        BS, xC, xH, xW = x.size()
        input=torch.cat((x,y),1)
        feature = self.inConv(input)
        resfeature=self.adConvList(feature)
        # feature=feature+resfeature
        feature = resfeature
        output=self.outConv(feature) + x
        return output



class ResNetBlock(nn.Module):
    def __init__(self):
        super(ResNetBlock, self).__init__()
        self.kernel_size = 3
        self.innerC = 64
        m = []
        m.append(nn.Conv2d(in_channels=self.innerC, out_channels=self.innerC, kernel_size=3, stride=1, padding=1,
                                bias=True))
        m.append(nn.ReLU())
        m.append(nn.Conv2d(in_channels=self.innerC, out_channels=self.innerC, kernel_size=3, stride=1, padding=1,
                                        bias=True))
        self.body = nn.Sequential(*m)
        # init_weights(self.body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.kernel_size = 3
        self.innerC = 64

        m.append(KNLConv(in_channel=self.innerC,kernel_size=self.kernel_size))
        m.append(nn.ReLU())
        m.append(KNLConv(in_channel=self.innerC, kernel_size=self.kernel_size))
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

