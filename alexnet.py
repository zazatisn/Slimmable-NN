import torch
import torch.nn as nn
import resnet9_v2

class AlexNet(resnet9_v2.ImageClassificationBase):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(64, 192, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), 256 * 3 * 3)
        logits = self.classifier(x)
        return logits
    
class SubAlexNet(resnet9_v2.ImageClassificationBase):
    def __init__(self, num_increments, subnetwork_ratio):
        super(SubAlexNet, self).__init__()
        conv1_kernels = int(64 * subnetwork_ratio) // num_increments
        conv2_kernels = int(192 * subnetwork_ratio) // num_increments
        conv3_kernels = int(384 * subnetwork_ratio) // num_increments
        conv4_kernels = int(256 * subnetwork_ratio) // num_increments
        conv5_kernels = int(256 * subnetwork_ratio) // num_increments
        linear1_size = int(1024 * subnetwork_ratio) // num_increments
        linear2_size = int(1024 * subnetwork_ratio) // num_increments

        self.features = nn.Sequential(
            nn.Conv2d(3, conv1_kernels, kernel_size=11, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(conv1_kernels, conv2_kernels, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(conv2_kernels, conv3_kernels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(conv3_kernels, conv4_kernels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(conv4_kernels, conv5_kernels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(conv5_kernels * 3 * 3, linear1_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(linear1_size, linear2_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        subnetwork_out = self.fc(x)
        return subnetwork_out
    
class SlimAlexNetIncrement(resnet9_v2.ImageClassificationBase):
    def __init__(self, num_increments, subnetwork_ratio, increment, subnets, num_classes):
        super().__init__()
        self.num_increments = num_increments
        self.subnets = subnets
        for i in range(increment):
            self.add_module(f'subnet{i}', subnets[i])
        self.increment = increment
        self.classifier = nn.Linear(increment * int(1024 * subnetwork_ratio) // num_increments, num_classes)

    def forward(self, x):
      subnet_xs = []
      for i in range(self.increment):
        subnet_xs.append(self.subnets[i](x))
      x = torch.cat(subnet_xs, dim=1)
      x = self.classifier(x)
      return x