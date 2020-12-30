import torch
import torch.nn as nn
import torch.nn.functional as F 

OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'skip_connect' : lambda C, stride, affine: Identity(),
  'sep_conv_3x3_spatial' : lambda C, stride, affine: SepConvAttention(C, C, 3, stride, 1, affine=affine),
  # 'sep_conv_5x5_spatial' : lambda C, stride, affine: SepConvAttention(C, C, 5, stride, 2, affine=affine),
  'dil_conv_3x3_spatial' : lambda C, stride, affine: DilConvAttention(C, C, 3, stride, 2, 2, affine=affine),
  #'dil_conv_5x5_spatial' : lambda C, stride, affine: DilConvAttention(C, C, 5, stride, 4, 2, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  #'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  #'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'SE':lambda C, stride, affine: SE_A(C, reduction=4),  # channel attention
  'SE_A_M':lambda C, stride, affine: SE_A_M(C, reduction=4),  # channel attention v2
  'CBAM':lambda C, stride, affine: CBAM(C, reduction_ratio=4),  
}

class ChannelPool(nn.Module):
    def forward(self, x):
        channel_out = torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
        #pdb.set_trace()
        return channel_out

class SepConvAttention(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConvAttention, self).__init__()
    self.compress = ChannelPool()
    C_in = 2
    C_out = 1
    self.op = nn.Sequential(
      
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    x_compress = self.compress(x)
    x_out = self.op(x_compress)
    scale = F.sigmoid(x_out)
    return x * scale

class DilConvAttention(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConvAttention, self).__init__()
    self.compress = ChannelPool()
    self.op = nn.Sequential(
      nn.Conv2d(2, 1, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1, bias=False),
      nn.Conv2d(1, 1, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(1, affine=affine),
      )

  def forward(self, x):
    x_compress = self.compress(x)
    x_out = self.op(x_compress)
    scale = F.sigmoid(x_out)
    return x * scale



class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      nn.Sigmoid(),
      )

  def forward(self, x):
    return self.op(x)*x


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      nn.Sigmoid()
      )

  def forward(self, x):
    return self.op(x)*x


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


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
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class SE_A(nn.Module):
  def __init__(self, channel, reduction=16):
    super(SE_A, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Sequential(
      nn.Linear(channel, channel // reduction, bias=False),
      nn.ReLU(inplace=False),
      nn.Linear(channel // reduction, channel, bias=False),
      nn.Sigmoid()
    )

  def forward(self, x):
    b, c, _, _ = x.size()
    y = self.avg_pool(x).view(b, c)
    y = self.fc(y).view(b, c, 1, 1)
    return x * y.expand_as(x)

class SE_A_M(nn.Module):
  def __init__(self, channel, reduction=16):
    super(SE_A_M, self).__init__()

    self.max_pool = nn.AdaptiveMaxPool2d(1)
    self.avg_pool = nn.AdaptiveAvgPool2d(1)

    self.fc = nn.Sequential(
      nn.Linear(channel, channel // reduction, bias=False),
      nn.ReLU(inplace=False),
      nn.Linear(channel // reduction, channel, bias=False),
    )

  def forward(self, x):
    b, c, _, _ = x.size()
    y_avg = self.avg_pool(x).view(b, c)
    y_avg = self.fc(y_avg)
    y_max = self.max_pool(x).view(b, c)
    y_max = self.fc(y_max)

    y =  y_max + y_avg 
    scale = F.sigmoid( y ).unsqueeze(2).unsqueeze(3).expand_as(x)
    return x * scale


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) 
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16,pool_types=['avg','max']):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out




