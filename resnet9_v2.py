import torch
import torch.nn as nn
import utility_functions
from utility_functions import ImageClassificationBase   
        
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 256, pool=True)
        self.res2 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(256, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

class SubResNet9(nn.Module):
    def __init__(self, in_channels, num_increments, subnetwork_ratio):
        super().__init__()
        conv1_out_channels = int(64 * subnetwork_ratio) // num_increments
        conv2_out_channels = int(128 * subnetwork_ratio) // num_increments
        conv3_out_channels = int(256 * subnetwork_ratio) // num_increments
        conv4_out_channels = int(256 * subnetwork_ratio) // num_increments
        
        self.conv1 = conv_block(in_channels, conv1_out_channels)
        self.conv2 = conv_block(conv1_out_channels, conv2_out_channels, pool=True)
        self.res1 = nn.Sequential(conv_block(conv2_out_channels, conv2_out_channels), conv_block(conv2_out_channels, conv2_out_channels))
        
        self.conv3 = conv_block(conv2_out_channels, conv3_out_channels, pool=True)
        self.conv4 = conv_block(conv3_out_channels, conv4_out_channels, pool=True)
        self.res2 = nn.Sequential(conv_block(conv4_out_channels, conv4_out_channels), conv_block(conv4_out_channels, conv4_out_channels))
        
        self.flatten = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), 
                                     nn.Flatten())
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.flatten(out)
        return out
    
class SlimResNet9Increment(ImageClassificationBase):
    def __init__(self, num_increments, subnetwork_ratio, increment, subnets, num_classes):
        super().__init__()
        self.num_increments = num_increments
        self.subnets = subnets
        for i in range(increment):
            self.add_module(f'subnet{i}', subnets[i])
        self.increment = increment
        self.classifier = nn.Sequential(nn.Dropout(0.2),
                                        nn.Linear(increment * int(256 * subnetwork_ratio) // num_increments, num_classes))

    def forward(self, x):
      subnet_xs = []
      for i in range(self.increment):
        subnet_xs.append(self.subnets[i](x))
      x = torch.cat(subnet_xs, dim=1)
      x = self.classifier(x)
      return x

