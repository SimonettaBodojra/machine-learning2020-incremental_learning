import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from libs.cifar100 import Cifar100, split_train_validation
from libs.resnet import resnet20, resnet32, resnet56
from sklearn.preprocessing import OneHotEncoder
import numpy as np

__one_hot_encoder = None
__class_map = []


# default arguments iCarl
def get_arguments():
    return {
        "LR": 1e-2,  # default iCarl 2
        "MOMENTUM": 0.9,
        "WEIGHT_DECAY": 5e-5, #1e-5
        "NUM_EPOCHS": 70,
        "MILESTONES": [49, 63],
        "BATCH_SIZE": 128,
        "DEVICE": 'cuda' if torch.cuda.is_available() else 'cpu',
        "GAMMA": 0.2,
        "SEED": 30,  # use 30, 42, 16, 1993
        "LOG_FREQUENCY": 30,
        "NUM_CLASSES": 100
    }


def get_train_eval_transforms():
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),  # Randomly flip the image with probability of 0.5
         transforms.Pad(4),
         transforms.RandomCrop(32),  # Crops a random squares of the image
         transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
         ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    return train_transform, eval_transform


def get_cifar_with_seed(root, transforms=None, src='train', seed=None):
    cifar = Cifar100(root, src, transforms)
    cifar.seed(seed)
    if src == 'train':
        init_one_hot_encoder(cifar)
    return cifar


def get_resnet(lr, momentum, weight_decay, milestones, gamma, resnet=32, loss_type='ce'):
    if resnet == 20:
        net = resnet20()
    elif resnet == 32:
        net = resnet32()
    elif resnet == 56:
        net = resnet56()
    else:
        raise ValueError("resnet parameter must be 20 32 or 56")

    if loss_type == 'ce':
      criterion = nn.CrossEntropyLoss()
    elif loss_type == 'bce':
      criterion = nn.BCEWithLogitsLoss(reduction='mean')
    else:
        raise ValueError("loss type must be 'bce' or 'ce'")
    parameters_to_optimize = net.parameters()
    optimizer = optim.SGD(parameters_to_optimize, lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma, last_epoch=-1)
    return net, criterion, optimizer, scheduler


def get_kth_batch(train_val_dataset: Cifar100, test_dataset: Cifar100, training_step: int, seed: int,
                  train_size=.8, get='indices'):
    if get not in ['indices', 'subsets']:
        raise ValueError("get parameter must be 'indices' or 'subsets'")

    train_val_classes = train_val_dataset.get_Kth_class_batch(training_step)
    test_classes = test_dataset.get_Kth_class_batch(training_step)

    if list(train_val_classes) != list(test_classes):
        raise SystemError("classes chosesn from training dataset and test dataset are not the same: " +
                          "probably a different seed has been set in datasets")

    train_idx, val_idx = split_train_validation(train_val_dataset, train_val_classes, train_size=train_size, seed=seed)
    idxs = test_dataset.get_item_idxs_of(test_classes, data_type='group')
    test_idx = []
    for idx in idxs:
        test_idx.extend(list(idx))

    if get == 'indices':
        return train_idx, val_idx, test_idx
    else:
        return Subset(train_val_dataset, train_idx), Subset(train_val_dataset, val_idx), Subset(test_dataset, test_idx)


def get_train_loader(dataset, batch_size=128):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)


def get_eval_loader(dataset, batch_size=128):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)


def init_one_hot_encoder(dataset: Cifar100):
    global __one_hot_encoder
    global __class_map
    __one_hot_encoder = OneHotEncoder(dtype=np.uint8)
    __class_map = list(dataset.int_to_class)
    labels = [[v, i] for i, v in enumerate(__class_map)]
    __one_hot_encoder = OneHotEncoder(dtype=np.uint8)
    __one_hot_encoder.fit(labels)


def one_hot_encode_labels(labels: torch.Tensor):
    global __class_map
    global __one_hot_encoder
    labels = labels.numpy()
    to_encode = [[__class_map[label], label] for label in labels]
    array = __one_hot_encoder.transform(to_encode).toarray()
    array = [np.array(v[:int(array.shape[1]/2)]) for v in array]
    array = np.array(array, dtype=np.float)
    return torch.from_numpy(array)
