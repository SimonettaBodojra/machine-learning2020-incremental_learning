import torch
import torch.nn as nn
from libs.resnet import resnet32
import copy


class LwfModel(nn.Module):

    def __init__(self, num_classes=10):
        super(LwfModel, self).__init__()
        self.known_classes = 0
        self.old_net = None
        self.net = resnet32(num_classes=num_classes)

    def after_train(self, class_batch_size):
        self.known_classes += class_batch_size
        self.old_net = copy.deepcopy(self.net)

    def increment_class(self, numclass):
        weight = self.net.fc.weight.data
        bias = self.net.fc.bias.data
        in_feature = self.net.fc.in_features
        out_feature = self.net.fc.out_features

        self.net.fc = nn.Linear(in_feature, numclass, bias=True)
        self.net.fc.weight.data[:out_feature] = weight
        self.net.fc.bias.data[:out_feature] = bias

    def forward(self, x, old_net=False):
        if old_net:
            return self.old_net(x)
        return self.net(x)


if __name__ == '__main__':
    ones = torch.ones(10)
    print(get_one_hot(ones, 5))