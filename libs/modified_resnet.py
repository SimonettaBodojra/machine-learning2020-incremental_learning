import math
import time
import random

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.model_zoo as model_zoo

from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.backends import cudnn
from itertools import chain

# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, last=False):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.last = last

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if not self.last:
            out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BiasLayer(nn.Module):

    def __init__(self, in_features, out_features):
        super(BiasLayer, self).__init__()

        # Layer dimensions
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))

        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True, device="cuda"))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True, device="cuda"))

    def forward(self, x):
        out = self.alpha * x + self.beta
        return out


class CosineLayer(nn.Module):

    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLayer, self).__init__()

        #Layer dimensions
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))

        # Sigma parameter
        self.sigma = Parameter(torch.Tensor(1)) if sigma else None

        # Reset layer parameter
        self.reset_parameters()

    def reset_parameters(self):

        std = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)

        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):

        # Compute output
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

        # Scale by sigma if set
        if self.sigma is not None:
            out = self.sigma * out

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, parameters, use_exemplars, classifier='fc', num_classes=10, k=5000):

        super(ResNet, self).__init__()

        self.classifier = classifier

        self.inplanes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        #Three layers composed of layers[i]-BasicBlock each (each BasicBlock has 2 layers)
        self.layer1 = self._make_layer(block, 16, layers[0])

        # a downsample layer (conv1x1) is added in the first BasicBlock
        # to adjust input to the output lower dimension caused by stride=2
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)

        if classifier == 'cosine':
            # a downsample layer (conv1x1) is added in the first BasicBlock to adjust
            # input to the output lower dimension caused by stride=2
            self.layer3 = self._make_layer(block, 64, layers[2], stride=2, last_phase=True)
        else:
            self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        #AVG pool on each feature map so output is size 1x1 with depth 64
        self.avgpool = nn.AvgPool2d(8, stride=1)

        if classifier == 'fc':
            self.fc = nn.Linear(64 * block.expansion, num_classes)
        elif classifier == 'cosine':
            self.fc = CosineLayer(64 * block.expansion, num_classes)
        elif classifier == 'bias':
            self.fc = nn.Linear(64 * block.expansion, num_classes)
            self.fc = BiasLayer(64 * block.expansion, num_classes)
        else:
            raise (ValueError('Unknown last layer type'))

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, last_phase=False):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        if last_phase:
            for i in range(1, blocks - 1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, last=True))
        else:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, output='all'):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        features = x.view(x.size(0), -1)
        out = self.fc(features)

        if output == 'features':
            return features
        elif output == 'fc':
            return out
        else:
            return out, features


