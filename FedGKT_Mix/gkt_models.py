import torch
import torch.nn as nn
import torch.nn.functional as F

#Norm selector
def Norm(planes, type, num_groups=2):
    if type == 'Batch Norm':
        return nn.BatchNorm2d(planes)
    elif type == 'Group Norm':
        return nn.GroupNorm(num_groups, planes)

#Bottleneck

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes,alpha_b, alpha_g , planes, stride=1, norm_type="Batch Norm"):
        super(Bottleneck, self).__init__()
        self.alpha_b = alpha_b
        self.alpha_g = alpha_g
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = Norm(planes, type="Batch Norm")
        self.gn1 = Norm(planes, type="Group Norm")
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn2 = Norm(planes, type="Batch Norm")
        self.gn2 = Norm(planes, type="Group Norm")
        self.conv3 = nn.Conv2d(planes, self.expansion *
                                planes, kernel_size=1, bias=False)
        self.bn3 = Norm(self.expansion * planes, type="Batch Norm")
        self.gn3 = Norm(self.expansion * planes, type="Group Norm")
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.alpha_b*self.bn1(self.conv1(x))+ self.alpha_g * self.gn1(self.conv1(x)))
        out = F.relu(self.alpha_b * self.bn2(self.conv2(out)) + self.alpha_g *self.gn2(self.conv2(out)))
        out = (self.alpha_b * self.bn3(self.conv3(out)) + self.alpha_g *self.gn3(self.conv3(out)))
        out += (self.alpha_b * self.bn3(self.shortcut(x)) + self.alpha_g *self.gn3(self.shortcut(x)))
        out = F.relu(out)
        return out

#Resnet-8
class ResNetClient(nn.Module):
    def __init__(self, block,alpha_b,alpha_g, num_blocks, num_classes=10, norm_type='Batch Norm'):
        super().__init__()
        self.in_planes = 16
        self.alpha_b = alpha_b
        self.alpha_g = alpha_g

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = Norm(self.in_planes, type="Batch Norm")
        self.gn1 = Norm(self.in_planes, type="Group Norm")
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.linear = nn.Linear(16*block.expansion, num_classes)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes,self.alpha_b,self.alpha_g, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.alpha_b*self.bn1(self.conv1(x))+ self.alpha_g * self.gn1(self.conv1(x)))
        extracted_features = out
        out = self.layer1(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, extracted_features

#Resnet-49
class ResNetServer(nn.Module):
    def __init__(self, block,alpha_b, alpha_g , num_blocks, num_classes=10, norm_type='Batch Norm'):
        super().__init__()
        
        self.alpha_b = alpha_b
        self.alpha_g = alpha_g
        self.in_planes = 16
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, norm_type=norm_type)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, norm_type=norm_type)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, norm_type=norm_type)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, norm_type=norm_type)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, norm_type):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes,self.alpha_b,self.alpha_g,planes, stride, norm_type))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet8(norm_type,alpha_b,alpha_g):
    return ResNetClient(Bottleneck,alpha_b,alpha_g, [2], norm_type=norm_type)


def ResNet49(norm_type,alpha_b,alpha_g):
    return ResNetServer(Bottleneck,alpha_b,alpha_g, [3, 4, 6, 3], norm_type=norm_type)