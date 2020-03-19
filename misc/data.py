import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

from collections import namedtuple

import torch
import torchvision

class DatasetBuilder(object):
    # tuple for dataset config
    DC = namedtuple('DatasetConfig', ['mean', 'std', 'input_size', 'num_classes'])
    
    DATASET_CONFIG = {
        'svhn' :       DC([0.43768210, 0.44376970, 0.47280442], [0.19803012, 0.20101562, 0.19703614], 32, 10),
        'cifar10':     DC([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784], 32, 10),
        'imagenet100': DC([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224, 100),
        'imagenet':    DC([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224, 1000),
    } 

    def __init__(self, name:str, root_path:str):
        """
        Args
        - name: name of dataset
        - root_path: root path to datasets
        """
        if name not in self.DATASET_CONFIG.keys():
            raise ValueError('name of dataset is invalid')
        self.name = name
        self.root_path = os.path.join(root_path, self.name)

    def __call__(self, train:bool, normalize:bool, binary_classification_target:int=None, optional_transform=[]):
        """
        Args
        - train : use train set or not.
        - normalize : do normalize or not.
        - binary_classification_target : if not None, creates datset for binary classification.
        """
        
        input_size = self.DATASET_CONFIG[self.name].input_size
        transform = self._get_transform(self.name, input_size, train, normalize, optional_transform)
        
        # get dataset
        if self.name == 'svhn':
            dataset = torchvision.datasets.SVHN(root=self.root_path, split='train' if train else 'test', transform=transform, download=True)
            targets_name = 'labels'
        elif self.name == 'cifar10':
            dataset = torchvision.datasets.CIFAR10(root=self.root_path, train=train, transform=transform, download=True)
            targets_name = 'targets'
        elif self.name in 'imagenet100 imagenet'.split():
            root = os.path.join(self.root_path, 'train' if train else 'val')
            dataset = torchvision.datasets.ImageFolder(root, transform=transform)
        else: 
            raise NotImplementedError 

        # make binary classification dataset
        if binary_classification_target is not None:
            targets = getattr(dataset, targets_name)
            assert binary_classification_target <= max(targets)

            targets = [1 if target==binary_classification_target else 0 for target in targets]
            setattr(dataset, targets_name, targets)

        return dataset

    def _get_transform(self, name:str, input_size:int, train:bool, normalize:bool, optional_transform=[]):
        """
        input_size
        - cifar10, svhn: 32x32
        - imagenet100, imagenet: 224x224  
        """
        transform = []

        # arugmentation
        # imagenet100 / imagenet
        if input_size == 224:
            if train:
                transform.extend([
                    torchvision.transforms.RandomResizedCrop(224),
				    torchvision.transforms.RandomHorizontalFlip(),
                ])
            else:
                transform.extend([
                    torchvision.transforms.Resize(256),
                    torchvision.transforms.CenterCrop(224),
                ])
        # cifar10 / svhn
        elif input_size == 32:
            if train:
                transform.extend([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomCrop(32, 4),
                ])
            else:
                pass
        else:
            raise NotImplementedError

        # to tensor
        transform.extend([torchvision.transforms.ToTensor(),])

        # normalize
        if normalize:
            transform.extend([
                torchvision.transforms.Normalize(mean=self.DATASET_CONFIG[name].mean, std=self.DATASET_CONFIG[name].std),
            ])

        # optional
        if optional_transform:
            transform.extend(optional_transform)

        return torchvision.transforms.Compose(transform)
    
    @property
    def input_size(self):
        return self.DATASET_CONFIG[self.name].input_size

    @property
    def num_classes(self):
        return self.DATASET_CONFIG[self.name].num_classes

if __name__ == '__main__':

    dataset_builder = DatasetBuilder(name='cifar10', root_path='../data')
    test_set = dataset_builder(train=False, normalize=True)
    print(test_set.targets)

    dataset_builder = DatasetBuilder(name='cifar10', root_path='../data')
    test_set = dataset_builder(train=False, normalize=True, binary_classification_target=7)
    print(test_set.targets)