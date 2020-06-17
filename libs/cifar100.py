import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from PIL import Image


def __unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='latin1')
    return cifar_dict


def _extract_cifar100(root, src='train'):
    if src not in ['train', 'test']:
        raise ValueError('src must be "train" or "test"')

    if src == 'train':
        number_images = 50000
    else:
        number_images = 10000

    meta = __unpickle(os.path.join(root, 'meta'))
    partition = __unpickle(os.path.join(root, src))
    classes = np.array(meta['fine_label_names'])

    images = partition['data']
    images = images.reshape(number_images, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
    labels = partition['fine_labels']

    return classes, images, labels


class Cifar100(VisionDataset):

    def __init__(self, root, src='train', transform=None):
        super(Cifar100, self).__init__(root, transform=transform)
        self.root = root
        self.transform = transform
        self.int_to_class, self.images, self.labels = _extract_cifar100(root, src)  # map int label to literal class
        self.class_to_int = {c: i for i, c in enumerate(self.int_to_class)}  # map literal class to int label

        self.df = pd.DataFrame({
            'image': pd.Series(list(self.images)),
            'label': self.labels,
        })

        self.splits = None
        self.__seed = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img, label = self.df.loc[index, 'image'], self.df.loc[index, 'label']

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)  # Return a PIL image

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __change_class_index(self):
        keys = list(self.int_to_class)
        np.random.seed(self.__seed)
        np.random.shuffle(keys)
        old_new_label_map = [0 for i in range(100)]
        for new_label, key in enumerate(keys):
            old_label = self.class_to_int[key]
            old_new_label_map[old_label] = new_label
            self.int_to_class[new_label] = key
            self.class_to_int[key] = new_label

        self.labels = [old_new_label_map[v] for v in self.labels]
        self.df = pd.DataFrame({
            'image': pd.Series(list(self.images)),
            'label': self.labels,
        })

    def __init_splits(self):
        if self.splits is None:
            """all_classes = list(self.class_to_int.values())
            np.random.seed(self.__seed)
            np.random.shuffle(all_classes)
            self.splits = [all_classes[start:start + 10] for start in range(0, 100, 10)]"""
            self.__change_class_index()
            self.splits = [np.arange(start, start + 10) for start in range(0, 100, 10)]

    def get_Kth_class_batch(self, step):
        if step < 0 or step > 9:
            raise ValueError('step must be between 0 and 9 included')

        self.__init_splits()
        return np.array(self.splits[step], dtype=np.uint8)

    def seed(self, seed):
        self.__seed = seed

    def clear_splits(self):
        self.splits = None

    def get_item_idxs_of(self, data, data_type='group'):  # group if data is a list of classes index, class if data
        if data_type not in ['class', 'group']:  # is a single class index
            raise ValueError('data_type must be "class" or "group"')
        if data_type == 'class':
            mask = self.df.loc[:, 'label'] == data
            idx = np.array(self.df.loc[mask].index)
            return idx
        else:
            idx_list = []
            for c in data:
                mask = self.df.loc[:, 'label'] == c
                idx_list.append(np.array(self.df.loc[mask].index))
            return idx_list

    def get_items_of(self, idxs):
        view = self.df.loc[idxs, :]
        return view.loc[:, 'image'], view.loc[:, 'label']


def split_train_validation(dataset: Cifar100, class_group, train_size=0.5, seed=None):
    from sklearn.model_selection import train_test_split
    train_idx = []
    val_idx = []
    idx_list = dataset.get_item_idxs_of(class_group, data_type='group')
    for idx in idx_list:
        t, v = train_test_split(idx, train_size=train_size, random_state=seed)
        train_idx.extend(list(t))
        val_idx.extend(list(v))

    return train_idx, val_idx


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


if __name__ == '__main__':
    import libs.utils as utils

    """test_dataset = utils.get_cifar_with_seed('../cifar-100-python', seed=42, src='train')
    group = cifar.get_Kth_class_batch(0)
    t, v = split_train_validation(cifar, group, seed=42)
    # print(len(set(map(lambda x: cifar[x][1], t))))
    # print(len(set(map(lambda x: cifar[x][1], v))))
    # print(cifar.get_items_of(t)[0].index)
    print(len(t))
    print(len(v))"""

    SEED = 42
    train_val_dataset = utils.get_cifar_with_seed('../cifar-100-python', seed=SEED, src='train')
    test_dataset = utils.get_cifar_with_seed('../cifar-100-python', seed=SEED, src='test')

    s1 = set()
    s2 = set()
    s3 = set()
    for stage in range(10):
        print(f"STARTING FINE TUNING STAGE {stage + 1}...")
        # Get indices
        # 4000 training, 1000 validation
        train_idx, val_idx, test_idx = utils.get_kth_batch(train_val_dataset, test_dataset, stage,
                                                           seed=SEED, train_size=.80, get='indices')

        _, l1 = train_val_dataset.get_items_of(train_idx)
        _, l2 = train_val_dataset.get_items_of(val_idx)
        _, l3 = test_dataset.get_items_of(test_idx)
        s1.update(l1)
        s2.update(l2)
        s3.update(l3)
        print(sorted(set(l1)))
        print(sorted(set(l2)))
        print(sorted(set(l3)))
        print()
    print(len(s1))
    print(len(s2))
    print(len(s3))

    import matplotlib.pyplot as plt
    img, lab = train_val_dataset[106]
    plt.figure(figsize=(.6, .6))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.show()
    print(train_val_dataset.int_to_class[lab])