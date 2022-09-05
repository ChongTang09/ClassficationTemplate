import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class AlexNet(nn.Module):
    def __init__(self, num_classes, spatial=True, pre_trained=True):
        super(AlexNet, self).__init__()

        __alexnet_arc = ['features', 'avgpool', 'classifier']

        model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=pre_trained)

        cnn = model.features
        cls = model.classifier

        self.features = nn.Sequential(
            cnn[0], # Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
            cnn[1], # ReLU(inplace=True)
            CBAM(64),
            cnn[2], # MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)

            cnn[3], # Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
            cnn[4], # ReLU(inplace=True)
            CBAM(192),
            cnn[5], # MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)

            cnn[6], # Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            cnn[7], # ReLU(inplace=True)
            CBAM(384),
            cnn[8], # Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            cnn[9], # ReLU(inplace=True)
            CBAM(256),
            cnn[10], # Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            cnn[11], # ReLU(inplace=True)
            CBAM(256),
            cnn[12], # MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        ) if spatial else cnn

        self.avgpool = model.avgpool

        self.classifier = nn.Sequential(
            cls[0], # Dropout(p=0.5, inplace=False)
            cls[1], # Linear(in_features=9216, out_features=4096, bias=True)
            cls[2], # ReLU(inplace=True)
            cls[3], # Dropout(p=0.5, inplace=False)
            cls[4], # Linear(in_features=4096, out_features=4096, bias=True)
            cls[5], # ReLU(inplace=True)
            nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        )

    def forward(self, x):
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out