from typing import Iterator

import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.nn import Parameter
from torch.utils.data import Dataset

import libs.utils as utils
from libs.resnet import resnet32
from libs.cifar100 import Cifar100
from libs.utils import get_one_hot
import copy


class ExemplarSet(Dataset):

    def __init__(self, images, labels, transforms):
        assert len(images) == len(labels)
        self.images = list(images)
        self.labels = list(labels)
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img, label = self.images[index], self.labels[index]

        img = Image.fromarray(img)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label


class iCaRLModel(nn.Module):

    def __init__(self, train_dataset: Cifar100, num_classes=100, memory=2000, batch_size=128, device='cuda'):
        super(iCaRLModel, self).__init__()
        self.num_classes = num_classes
        self.memory = memory
        self.known_classes = 0
        self.old_net = None
        self.batch_size = batch_size
        self.device = device

        self.net = resnet32(num_classes=num_classes)
        self.dataset = train_dataset

        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.exemplar_sets = []

        self.compute_means = True
        self.exemplar_means = []

    def forward(self, x):
        return self.net(x)

    def _extract_features(self, images):
        return self.net(images, features=True)

    def increment_known_classes(self, n_new_classes=10):
        self.known_classes += n_new_classes

    def combine_trainset_exemplars(self, train_dataset: Cifar100):
        exemplar_indexes = np.hstack(self.exemplar_sets)
        images, labels = self.dataset.get_items_of(exemplar_indexes)
        exemplar_dataset = ExemplarSet(images, labels, utils.get_train_eval_transforms()[1])
        return utils.create_augmented_dataset(train_dataset, exemplar_dataset)

    def update_representation(self, train_dataset: Cifar100, optimizer, scheduler, num_epochs):
        self.compute_means = True
        self.net = self.net.to(self.device)

        if len(self.exemplar_sets) > 0:
            self.old_net = copy.deepcopy(self.net)
            self.old_net = self.old_net.to(self.device)
            train_dataset = self.combine_trainset_exemplars(train_dataset)

        loader = utils.get_train_loader(train_dataset, self.batch_size, drop_last=False)

        train_losses = []
        train_accuracies = []

        for epoch in range(num_epochs):
            print(f"\tSTARTING EPOCH {epoch + 1} - LR={scheduler.get_last_lr()}...")
            cumulative_loss = .0
            running_corrects = 0
            self.net.train()
            for i, (images, labels) in enumerate(loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.net(images)

                _, preds = torch.max(outputs.data, 1)
                running_corrects += torch.sum(preds == labels.data).data.item()

                loss = self.compute_distillation_loss(images, labels, outputs)
                loss_value = loss.item()
                cumulative_loss += loss_value

                loss.backward()
                optimizer.step()

                if i != 0 and i % 20 == 0:
                    print(f"\t\tEpoch {epoch + 1}: Train_loss = {loss_value}")

            curr_train_loss = cumulative_loss / float(len(train_dataset))
            curr_train_accuracy = running_corrects / float(len(train_dataset))
            train_losses.append(curr_train_loss)
            train_accuracies.append(curr_train_accuracy)
            scheduler.step()

            print(f"\t\tRESULT EPOCH {epoch + 1}:")
            print(f"\t\t\tTrain Loss: {curr_train_loss} - Train Accuracy: {curr_train_accuracy}\n")

        return np.mean(train_losses), np.mean(train_accuracies)

    def compute_distillation_loss(self, images, labels, new_outputs):

        if self.known_classes == 0:
            return self.bce_loss(new_outputs, get_one_hot(labels, self.num_classes, self.device))

        sigmoid = nn.Sigmoid()
        n_old_classes = self.known_classes
        old_outputs = self.old_net(images)

        targets = get_one_hot(labels, self.num_classes, self.device)
        targets[:, :n_old_classes] = sigmoid(old_outputs)[:, :n_old_classes]
        tot_loss = self.bce_loss(new_outputs, targets)

        return tot_loss

    def classify(self, images, method='nearest-mean'):
        if method == 'nearest-mean':
            return self._nme(images)
        elif method == 'fc':
            outputs = self.net(images)
            _, preds = torch.max(outputs.data, 1)
            return preds

    def _nme(self, images):
        if self.compute_means:
            exemplar_means = []
            for exemplar_class_idx in self.exemplar_sets:
                imgs, labs = self.dataset.get_items_of(exemplar_class_idx)
                exemplars = ExemplarSet(imgs, labs, utils.get_train_eval_transforms()[1])
                loader = utils.get_eval_loader(exemplars, self.batch_size)
                self.net.eval()
                flatten_features = []
                with torch.no_grad():
                    for imgs, _ in loader:
                        imgs = imgs.to(self.device)
                        features = self._extract_features(imgs)
                        flatten_features.append(features)

                    flatten_features = torch.cat(flatten_features).to(self.device)
                    class_mean = flatten_features.mean(0)
                    class_mean = class_mean / class_mean.norm()
                    exemplar_means.append(class_mean)

            self.compute_means = False
            self.exemplar_means = exemplar_means

        exemplar_means = self.exemplar_means
        means = torch.stack(exemplar_means)  # (n_classes, feature_size)
        means = torch.stack([means] * len(images))  # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2)  # (batch_size, feature_size, n_classes)

        feature = self._extract_features(images)
        feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
        feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)

        dists = (feature - means).pow(2).sum(1).squeeze()  # (batch_size, n_classes)
        _, preds = dists.min(1)

        return preds

    def reduce_exemplar_set(self, m, label):
        # for i, exemplar_set in enumerate(self.exemplar_sets):
        self.exemplar_sets[label] = self.exemplar_sets[label][:m]

    def construct_exemplar_set(self, indexes, images, label, m, herding=True):
        if herding:
            self.herding_construct_exemplar_set(indexes, images, label, m)
        else:
            pass

    def herding_construct_exemplar_set(self, indexes, images, label, m):
        exemplar_set = ExemplarSet(images, [label] * len(images), utils.get_train_eval_transforms()[1])
        loader = utils.get_eval_loader(exemplar_set, self.batch_size)

        self.net.eval()
        flatten_features = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(self.device)
                features = self._extract_features(images)
                flatten_features.append(features)

            flatten_features = torch.cat(flatten_features).cpu().numpy()
            class_mean = np.mean(flatten_features, axis=0)
            class_mean = class_mean / np.linalg.norm(class_mean)
            # class_mean = torch.from_numpy(class_mean).to(self.device)
            flatten_features = torch.from_numpy(flatten_features).to(self.device)

        exemplars = set()  # lista di exemplars selezionati per la classe corrente
        exemplar_feature = []  # lista di features per ogni exemplars già selezionato
        for k in range(m):
            S = 0 if k == 0 else torch.stack(exemplar_feature).sum(0)
            phi = flatten_features
            mu = class_mean
            mu_p = ((phi + S) / (k + 1)).cpu().numpy()
            mu_p = mu_p / np.linalg.norm(mu_p)
            distances = np.sqrt(np.sum((mu - mu_p) ** 2, axis=1))
            # Evito che si creino duplicati
            sorted_indexes = np.argsort(distances)
            for i in sorted_indexes:
                if indexes[i] not in exemplars:
                    exemplars.add(indexes[i])
                    exemplar_feature.append(flatten_features[i])
                    break

        assert len(exemplars) == m
        self.exemplar_sets.append(list(exemplars))

    def parameters(self, recurse: bool = ...) -> Iterator[Parameter]:
        return self.net.parameters()
