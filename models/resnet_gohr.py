import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(in_planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.shortcut = nn.Sequential()
        # if in_planes != planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv1d(in_planes, planes, kernel_size=1),
        #         nn.BatchNorm2d(self.expansion*planes)
        #     )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv1d(4, 32, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(32)

        self.layers = []
        for nb in num_blocks:
            self.layers.append(self._make_layer(block, 32, nb))

        self.linear1 = nn.Linear(512, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 1)

    def _make_layer(self, block, planes, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):

        x = x.view(x.shape[0], 4, 16)
        out = F.relu(self.bn1(self.conv1(x)))

        for i in range(len(self.layers)):
            out = self.layers[i](out)
        
        feature = out.view(out.size(0), -1)
        out = self.linear1(feature)
        out = self.linear2(feature)
        out = self.linear3(feature)
        return out

def ResNet_Gohr(blocks=10):
    return ResNet(BasicBlock, [2] * blocks)