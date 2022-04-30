'''FPN in PyTorch.
See the paper "Feature Pyramid Networks for Object Detection" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from mmdet.models.builder import NECKS
from mmcv.runner import BaseModule

class conv_dy(BaseModule):
    def __init__(self, inplanes, planes, kernel_size, stride, padding, bias=False):
        super(conv_dy,self).__init__()
        K = 4
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  

        self.fc = nn.Sequential(
            nn.Linear(inplanes, int(inplanes/4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(inplanes/4), K, bias=False),
        ) 
        self.softmax = nn.Softmax(dim = 1)
        for i in range(1,K+1,1):
            setattr(self, 'conv2d_'+str(i), nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=kernel_size, stride=stride, padding=padding,bias=bias))
 
    def forward(self, x):
        t = 30
        # 计算权重 t是退火的温度
        pi = self.avg_pool(x)
        batch, c,_,_=pi.size()
        pi = pi.reshape(batch,-1)
        pi = self.fc(pi)/t
        pi = self.softmax(pi)
        h,w = pi.size()
        # 不同的权重计算的卷积结果
        y1 = self.conv2d_1(x)
        y2 = self.conv2d_2(x)
        y3 = self.conv2d_2(x)
        y4 = self.conv2d_2(x)


        # 计算加权和
        y = y1*pi[:,0].expand(1,1,1,h).reshape(h,1,1,1)\
            +y2*pi[:,1].expand(1,1,1,h).reshape(h,1,1,1)\
            +y3*pi[:,2].expand(1,1,1,h).reshape(h,1,1,1)\
            +y4*pi[:,3].expand(1,1,1,h).reshape(h,1,1,1)
        return y


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv_dy(in_planes, planes, kernel_size=1, stride = 1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_dy(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv_dy(planes, self.expansion*planes, kernel_size=1, stride = 1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

@NECKS.register_module()
class FPN_dcd(BaseModule):
    def __init__(self, *args, **kwds):
        super(FPN_dcd, self).__init__()
        '''self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        num_blocks = [2,2,2,2]
        # Bottom-up layers
        self.layer1 = self._make_layer(  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer( 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer( 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer( 512, num_blocks[3], stride=2)'''

        # Top layer
        self.toplayer = conv_dy(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = conv_dy(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = conv_dy(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = conv_dy(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = conv_dy(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = conv_dy( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)
        
        #extra_layers
        self.extra1 = conv_dy(2048, 256, kernel_size=3, stride=2, padding=1)
        self.extra2 = conv_dy(2048, 256, kernel_size=3, stride=2, padding=1)

    '''def _make_layer(self,planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Bottleneck(self.in_planes, planes, stride))
            self.in_planes = planes * Bottleneck.expansion
        return nn.Sequential(*layers)'''

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        
        #laterals_layer
        c6 = self.toplayer(x[3])
        p5 = self.smooth1(c6)
        c5 = self._upsample_add(c6, self.latlayer1(x[2]))
        p4 = self.smooth2(c5)
        c4 = self._upsample_add(c5, self.latlayer2(x[1]))
        p3 = self.smooth3(c4)
        #extra_layer
        p2 = self.extra1(x[3])
        p1 = self.extra2(x[3])

        return (p1, p2, p3, p4, p5)


"""def FPN101():
    # return FPN(Bottleneck, [2,4,23,3])
    return FPN_dcd()
def test():
    net = FPN101()
    fms = net((Variable(torch.randn(1,256,600,600)),\
        Variable(torch.randn(1,512,300,300)),Variable(torch.randn(1,1024,150,150)),Variable(torch.randn(1,2048,75,75))))
    for fm in fms:
        print(fm.size())
test()"""

