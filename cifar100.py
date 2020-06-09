import os
import pandas as pd
import numpy as np
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
            'image':  pd.Series(list(self.images)),
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

    def __init_splits(self):
        if self.splits is None:
            all_classes = list(self.class_to_int.values())
            np.random.seed(self.__seed)
            np.random.shuffle(all_classes)
            self.splits = [all_classes[start:start+10] for start in range(10)]

    def get_Kth_batch(self, step):
        if step < 0 or step > 9:
            raise ValueError('step must be between 0 and 9 included')

        self.__init_splits()
        return np.array(self.splits[step], dtype=np.uint8)

    def seed(self, seed):
        self.__seed = seed

    def clear_splits(self):
        self.splits = None

    def get_img_idxs_of(self, data, data_type='class'):
        if data_type not in ['class', 'group']:
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


def split_train_validation(dataset: Cifar100, class_group, train_size=0.5, seed=None):
    from sklearn.model_selection import train_test_split
    train_idx = []
    val_idx = []
    idx_list = dataset.get_img_idxs_of(class_group, data_type='group')
    for idx in idx_list:
        t, v = train_test_split(idx, train_size=train_size, random_state=seed)
        train_idx += list(t)
        val_idx += list(v)

    return train_idx, val_idx


if __name__ == '__main__':
    cifar = Cifar100('cifar-100-python')
    cifar.seed(42)
    group = cifar.get_Kth_batch(0)
    t, v = split_train_validation(cifar, group, seed=42)
    print(len(set(map(lambda x: cifar[x][1], t))))
    print(len(set(map(lambda x: cifar[x][1], v))))
