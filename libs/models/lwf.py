from typing import Iterator

import torch.nn as nn
from torch.nn import Parameter

from libs.resnet import resnet32
import copy
from libs.utils import get_one_hot


class LwfModel(nn.Module):

    def __init__(self, num_classes=10):
        super(LwfModel, self).__init__()
        self.num_classes = num_classes
        self.known_classes = 0
        self.old_net = None
        self.net = resnet32(num_classes=num_classes)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def before_train(self, device):
        self.net.to(device)
        if self.old_net is not None:
            self.old_net.to(device)
            self.old_net.eval()

    def after_train(self, class_batch_size):
        self.known_classes += class_batch_size
        self.old_net = copy.deepcopy(self)

    def increment_class(self, num_classes=10):
        weight = self.net.fc.weight.data
        bias = self.net.fc.bias.data
        in_feature = self.net.fc.in_features
        out_feature = self.net.fc.out_features

        self.net.fc = nn.Linear(in_feature, num_classes, bias=True)
        self.net.fc.weight.data[:out_feature] = weight
        self.net.fc.bias.data[:out_feature] = bias

    def forward(self, x):
        return self.net(x)

    def compute_distillation_loss(self, images, labels, new_outputs, device, num_classes=10):

        if self.known_classes == 0:
            return self.bce_loss(new_outputs, get_one_hot(labels, self.num_classes, device))

        sigmoid = nn.Sigmoid()
        n_old_classes = self.known_classes
        # n_new_classes = self.known_classes + num_classes
        old_outputs = self.old_net(images)

        targets = get_one_hot(labels, self.num_classes, device)
        targets[:, :n_old_classes] = sigmoid(old_outputs)[:, :n_old_classes]
        tot_loss = self.bce_loss(new_outputs, targets)

        return tot_loss

    def parameters(self, recurse: bool = ...) -> Iterator[Parameter]:
        return self.net.parameters()
