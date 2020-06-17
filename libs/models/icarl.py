import copy
from typing import Iterator

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.utils.data import Dataset

import libs.utils as utils
from libs.resnet import resnet32
from libs.utils import get_one_hot
import numpy as np


class iCaRLModel(nn.Module):

    def __init__(self, num_classes=100, memory=2000):
        super(iCaRLModel, self).__init__()
        self.num_classes = num_classes
        self.memory = memory
        self.known_classes = 0
        self.old_net = None

        self.net = resnet32(num_classes=num_classes)

        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.exemplar_sets = [{'indexes': [], 'features': []} for label in range(0, num_classes)]

    def before_train(self, device):
        self.net.to(device)
        if self.old_net is not None:
            self.old_net.to(device)
            self.old_net.eval()

        indexes = [diz['indexes'] for diz in self.exemplar_sets[:self.known_classes]]
        return np.ravel(indexes)

    def after_train(self, class_batch_size, train_subsets_per_class, labels, device, herding=True):
        self.known_classes += class_batch_size
        self.old_net = copy.deepcopy(self)

        self.net = self.net.to(device)

        min_memory = self.memory / self.known_classes
        class_memories = [min_memory] * self.known_classes
        empty_memory = self.memory % self.known_classes
        if empty_memory > 0:
            for i in range(empty_memory):
                class_memories[i] += 1

        assert sum(class_memories) == 2000

        for i, m in enumerate(class_memories[: self.known_classes - class_batch_size]):
            self.reduce_exemplar_set(m, i)

        for curr_subset, label, m in zip(train_subsets_per_class, labels,
                                         class_memories[self.known_classes - class_batch_size: self.known_classes]):
            self.construct_exemplar_set(curr_subset, label, m, device, herding=herding)

    def increment_class(self, num_classes=10):
        weight = self.net.fc.weight.data
        bias = self.net.fc.bias.data
        in_feature = self.net.fc.in_features
        out_feature = self.net.fc.out_features

        self.net.fc = nn.Linear(in_feature, num_classes, bias=True)
        self.net.fc.weight.data[:out_feature] = weight
        self.net.fc.bias.data[:out_feature] = bias

    def forward(self, x, features=False):
        return self.net(x, features)

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

    def _extract_feature(self, x):
        return self.net(x, features=True)

    def construct_exemplar_set(self, single_class_dataset, label, m, device, herding=True):
        if len(single_class_dataset) < m:
            raise ValueError("Number of images can't be less than m")

        if herding:
            loader = utils.get_eval_loader(single_class_dataset, batch_size=256)
            features = []
            map_subset_to_cifar = single_class_dataset.indices

            self.net.eval()
            with torch.no_grad():
                for images, _ in loader:
                    images = images.to(device)
                    feat = self._extract_feature(images)
                    features.append(feat)

                flatten_features = torch.cat(features)
                class_mean = flatten_features.mean(0)
                class_mean = class_mean / class_mean.norm(p=2)

            for k in range(m):
                min_index = -1
                min_dist = .0
                exemplars = self.exemplar_sets[label]['index']
                for i, feature in enumerate(flatten_features):
                    if i in exemplars:
                        continue
                    sum_exemplars = 0 if k == 0 else sum(exemplars[:k])
                    sum_exemplars = (feature + sum_exemplars) / (k + 1)
                    curr_mean_exemplars = sum_exemplars / sum_exemplars.norm(p=2)
                    curr_dist = torch.dist(class_mean, curr_mean_exemplars)

                    if min_index == -1 or min_dist > curr_dist:
                        min_index = i
                        min_dist = curr_dist

                self.exemplar_sets[label]['indexes'].append(map_subset_to_cifar[min_index])
                self.exemplar_sets[label]['features'].append(flatten_features[min_index])

    def reduce_exemplar_set(self, m, label):
        if len(self.exemplar_sets[label]) < m:
            raise ValueError(f"m must be lower than current size of current exemplar set for class {label}")

        self.exemplar_sets[label]['indexes'] = self.exemplar_sets[label]['indexes'][:m]
        self.exemplar_sets[label]['features'] = self.exemplar_sets[label]['features'][:m]


class AugmentedDataset(Dataset):

    def __init__(self, new_class_dataset, old_class_exemplars):
        self.new_class_dataset = new_class_dataset
        self.old_class_dataset = old_class_exemplars
        self.l1 = len(new_class_dataset)
        self.l2 = len(old_class_exemplars)

    def __getitem__(self, index):
        if index < self.l1:
            return self.new_class_dataset[index]  # here it leans on cifar100 get item
        else:
            return self.old_class_dataset[index - self.l1]

    def __len__(self):
        return self.l1 + self.l2
