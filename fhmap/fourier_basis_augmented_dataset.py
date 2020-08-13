import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import random
import numpy as np
import torch
import torchvision

from fhmap.fourier_heatmap import AddFourierNoise


class FourierBasisAugmentedDataset():
    def __init__(self, basedataset, input_size, mean, std, h_index, w_index, eps, randomize_index: bool = True, normalize: bool = True, mode: str = 'freq'):
        """
        Args:
        - basedataset
        - randomize_index
        """

        self.basedataset = basedataset
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.h_index = -h_index if h_index > 0 else h_index
        self.w_index = -w_index if (self.h_index == 0) and (w_index > 0) else w_index
        self.eps = eps
        self.randomize_index = randomize_index
        self.normalize = normalize
        self.mode = mode

        assert -int(np.floor(self.input_size / 2.0)) <= self.h_index <= 0
        assert -int(np.floor(self.input_size / 2.0)) <= self.w_index <= int(np.floor(self.input_size / 2.0))

    def __getitem__(self, index):
        x, t = self.basedataset[index]
        c, h, w = x.shape[-3:]

        if self.randomize_index:
            h_index = random.randrange(self.h_index, 1)
            if h_index == 0:
                if self.w_index >= 0:
                    w_index = random.randrange(-self.w_index, 1)
                else:
                    w_index = random.randrange(self.w_index, 1)
            else:
                if self.w_index >= 0:
                    w_index = random.randrange(-self.w_index, self.w_index + 1)
                else:
                    w_index = random.randrange(self.w_index, -self.w_index + 1)
        else:
            h_index = self.h_index
            w_index = self.w_index

        x = AddFourierNoise(h_index, w_index, eps=self.eps, norm_type='l2')(x)

        if self.normalize:
            x = torchvision.transforms.Normalize(mean=self.mean, std=self.std)(x)

        t_fb = self._get_target_label(h_index, w_index, self.mode)

        return x, t, t_fb

    def __len__(self):
        return len(self.basedataset)

    def _get_target_label(self, h_index, w_index, mode: str):
        if mode == 'freq':
            label = int(np.sqrt((h_index * h_index) + (w_index * w_index)))
        elif mode == 'index':
            half_input_size = int((self.input_size - 1) / 2.0) if (self.input_size % 2) == 0 else int(self.input_size / 2.0)
            label = int(((h_index + half_input_size) * self.input_size) + (w_index + half_input_size))
        else:
            raise NotImplementedError

        return label


if __name__ == '__main__':
    import os
    import sys
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
    sys.path.append(base)
    from misc.data import DatasetBuilder

    dataset_builder = DatasetBuilder(name='cifar10', root_path='../data')
    dataset = dataset_builder(train=True, normalize=False)
    print(type(dataset))
    mean = [0.49139968, 0.48215841, 0.44653091]
    std = [0.24703223, 0.24348513, 0.26158784]

    for ih in range(-15, 1):
        for iw in range(-15, 16):
            fbaug_dataset = FourierBasisAugmentedDataset(dataset, input_size=32, mean=mean, std=std, h_index=ih, w_index=iw, eps=4.0, randomize_index=False, normalize=False, mode='index')

            loader = torch.utils.data.DataLoader(fbaug_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)

            for i, (x, t, t_fb) in enumerate(loader):
        
                # print(x.shape)
                # print(t)
                print('{ih}, {iw}'.format(ih=ih, iw=iw))
                print(t_fb)
                torchvision.utils.save_image(x, '../logs/sample_fbaug.png')
                break