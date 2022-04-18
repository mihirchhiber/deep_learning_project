import torch
import torch.nn as nn

def conv(ni, nf, ks=3, stride=1, pad=1, bias=False):
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=pad, bias=bias)

def conv_layer(ni, nf, ks=3, stride=1, pad=1,act=True):
    bn = nn.BatchNorm2d(nf)
    layers = [conv(ni, nf, ks, stride, pad), bn]
    act_fn = nn.ReLU(inplace=True)
    if act: layers.append(act_fn)
    return nn.Sequential(*layers)

class InceptionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1x1 = conv_layer(256*2, 64*2, ks=1, stride=1, pad=0)
        
        self.branch1x1_pool = conv_layer(256*2, 32*2, ks=1, stride=1, pad=0)
        self.branch1x1_final = conv_layer(32*2, 32*2, ks=1, stride=1, pad=0)
        
        self.branch3x3_init = conv_layer(256*2, 128*2, ks=3, stride=1, pad=1)
        self.branch3x3_final = conv_layer(128*2, 128*2, ks=3, stride=1, pad=1)
        
        self.branch5x5_init = conv_layer(256*2, 32*2, ks=5, stride=1, pad=2)
        self.branch5x5_final = conv_layer(32*2, 32*2, ks=5, stride=1, pad=2)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        x1 = self.branch1x1(x)
        
        x2 = self.branch1x1_pool(x)
        x3 = self.branch1x1_final(x2)
        
        x4 = self.branch3x3_init(x)
        x5 = self.branch3x3_final(x4)
        
        x6 = self.branch5x5_init(x)
        x7 = self.branch5x5_final(x6)
        
        ans = [x1,x3,x5,x7]
        return self.relu(torch.cat(ans,1))

def conv_layer_averpl(ni, nf):
    aver_pl = nn.AvgPool2d(kernel_size=2, stride=2)
    return nn.Sequential(conv_layer(ni, nf), aver_pl)