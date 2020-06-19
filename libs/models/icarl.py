import copy
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter

import libs.utils as utils
from libs.resnet import resnet32
from libs.utils import get_one_hot


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
        return [] if len(indexes) == 0 else np.hstack(indexes)

    def after_train(self, class_batch_size, train_subsets_per_class, labels, device, herding=True):
        self.known_classes += class_batch_size
        self.old_net = copy.deepcopy(self)

        self.net = self.net.to(device)

        min_memory = int(self.memory / self.known_classes)
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
            print(label)
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

    def compute_exemplars_means(self, device):
        means = []
        for diz in self.exemplar_sets[:self.known_classes]:
            features = torch.stack(diz['features']).to(device)
            class_mean = features.mean(0)
            class_mean = class_mean / class_mean.norm()
            means.append(class_mean)

        return torch.stack(means).to(device)

    def classify(self, images, device, method='nearest-mean'):
        if method == 'nearest-mean':
            return self._nearest_mean(images, device)
        elif method == 'fc':
            outputs = self.net(images)
            _, preds = torch.max(outputs.data, 1)
            return preds
        elif method == 'knn':
            return self._k_nearest_neighbours(images)
        else:
            raise ValueError("method must be one of 'nearest-mean', 'fc', 'knn', 'svm'")

    def _nearest_mean(self, images, device):
        means = self.compute_exemplars_means(device)
        targets = torch.zeros(len(images), dtype=torch.int).to(device)

        self.net.eval()
        with torch.no_grad():
            features = self._extract_feature(images)
            for i, feat in enumerate(features):
                dists = torch.stack([torch.dist(feat, mean) for mean in means]).to(device)
                targets[i] = int(torch.argmin(dists))

        return targets.to(device)

    def _k_nearest_neighbours(self, images):
        self.net.eval()

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

        map_subset_to_cifar = np.array(single_class_dataset.indices)
        loader = utils.get_eval_loader(single_class_dataset, batch_size=256)
        features = []
        if herding:
            self.net.eval()
            with torch.no_grad():
                for images, _ in loader:
                    images = images.to(device)
                    feat = self._extract_feature(images)
                    features.append(feat)

                flatten_features = torch.cat(features).to(device)
                class_mean = flatten_features.mean(0)
                class_mean = (class_mean / class_mean.norm()).to(device)

            exemplars_indexes = set()
            for k in range(m):
                exemplars = self.exemplar_sets[label]['features']

                if len(exemplars) > 0:
                    exemplars = torch.stack(exemplars).to(device)

                sum_exemplars = 0 if k == 0 else exemplars.sum(0)
                mean_exemplars = (flatten_features.to(device) + sum_exemplars) / (k + 1)
                mean_exemplars = torch.stack([torch.dist(class_mean.to(device), e_sum) for e_sum in mean_exemplars])
                idxs = torch.argsort(mean_exemplars)
                min_index = 0
                for i in idxs:
                    i = int(i)
                    if i not in exemplars_indexes:
                        exemplars_indexes.add(i)
                        min_index = i
                        break

                self.exemplar_sets[label]['indexes'].append(map_subset_to_cifar[min_index])
                self.exemplar_sets[label]['features'].append(flatten_features[min_index].cpu())

        else:
            self.net.eval()
            indexes = []
            with torch.no_grad():
                for i, (images, _) in enumerate(loader):
                    choices = np.arange(len(images))
                    samples = np.random.choice(choices, m, replace=False)
                    images = images[samples].to(device)
                    feat = self._extract_feature(images)
                    features.append(feat)
                    curr_idx = (i * 256) + samples
                    curr_idx = [map_subset_to_cifar[i] for i in curr_idx]
                    indexes.extend(curr_idx)

                flatten_features = torch.cat(features)
                self.exemplar_sets[label]['indexes'] = indexes
                self.exemplar_sets[label]['features'] = list(flatten_features)

    def reduce_exemplar_set(self, m, label):
        if len(self.exemplar_sets[label]['indexes']) < m:
            raise ValueError(f"m must be lower than current size of current exemplar set for class {label}")

        self.exemplar_sets[label]['indexes'] = self.exemplar_sets[label]['indexes'][:m]
        self.exemplar_sets[label]['features'] = self.exemplar_sets[label]['features'][:m]
