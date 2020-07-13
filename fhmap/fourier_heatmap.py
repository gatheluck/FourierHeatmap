import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import tqdm
import random
import collections
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns

from misc.metric import get_num_correct
from fhmap.fourier_base import generate_fourier_base


class AddFourierNoise(object):
    def __init__(self, h_index: int, w_index: int, eps: float, norm_type: str = 'l2'):
        """
        Add Fourier noise to RGB channels respectively.
        This class is able to use as same as the functions in torchvision.transforms.

        Args
            h_index: index of fourier basis about hight direction
            w_index: index of fourier basis about width direction
            eps: size of noise(perturbation)
        """
        assert eps >= 0.0
        self.h_index = h_index
        self.w_index = w_index
        self.eps = eps
        self.norm_type = norm_type

    def __call__(self, x: torch.tensor):
        c, h, w = x.shape[-3:]
        assert c == 3
        assert h >= 1 or w >= 1
        assert abs(self.h_index) <= np.floor(h / 2) and abs(self.w_index) <= np.floor(w / 2)

        fourier_base = generate_fourier_base(h, w, self.h_index, self.w_index)  # l2 normalized fourier base

        # if self.norm_type == 'linf':
        #     fourier_base /= fourier_base.abs().max()
        #     fourier_base *= self.eps / 255.0
        # elif self.norm_type == 'l2':
        #     fourier_base /= fourier_base.norm()
        #     eps_l2 = np.sqrt(((self.eps / 255.0)**2.0) * h * w)
        #     fourier_base *= eps_l2
        # else:
        #     raise NotImplementedError
        fourier_base /= fourier_base.norm()
        fourier_base *= self.eps / 255.0

        fourier_noise = fourier_base.unsqueeze(0).repeat(c, 1, 1)  # (c, h, w)

        # multiple random noise form [-1, 1]
        fourier_noise[0, :, :] *= random.randrange(-1, 2, 2)
        fourier_noise[1, :, :] *= random.randrange(-1, 2, 2)
        fourier_noise[2, :, :] *= random.randrange(-1, 2, 2)

        return torch.clamp(x + fourier_noise, min=0.0, max=1.0)


def create_fourier_heatmap(model, dataset_builder, h_map_size: int, w_map_size: int, eps: float, norm_type: str, num_samples: int, batch_size: int, num_workers: int, log_dir: str, top_k: int = 1, suffix: str = '', shuffle: bool = False, orator: bool = False, **kwargs):
    """
    Args
    - model: NN model
    - dataset_builder: DatasetBuilder class object
    - h_map_size: height of Fourier heatmap
    - w_map_size: width of Fourier heatmap
    - eps: perturbation size
    - norm_type: type of norm to normalize Fourier basis
    - num_samples: number of samples. if -1, use all samples
    - batch_size: size of batch
    - num_workers: number of workers
    - top_k: use top_k accuracy to compute Fourier heatmap
    - log_dir: log directory
    - suffix: suffix of log
    - shuffle: shuffle dataset
    - orator: if True, show current result
    """

    assert (h_map_size % 2 == 1) and (h_map_size > 0), 'h_map_size should be odd because of symmetry'
    assert (w_map_size % 2 == 1) and (w_map_size > 0), 'w_map_size should be odd because of symmetry'
    assert eps > 0.0, 'eps should be larger than 0.0'
    assert norm_type in 'linf l2'.split(), 'value of norm_type option is invalid.'
    assert (num_samples > 0) or (num_samples == -1), 'num_samples should be larger than 0'
    assert batch_size > 0, 'batch_size should be larger than 0'
    assert num_workers > 0, 'num_workers should be larger than 0'
    assert top_k > 0, 'top_k should be larger than 0'

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Fourier Heatmap
    error_matrix = torch.zeros(h_map_size, w_map_size).float()
    images_list_former_half = collections.deque()
    images_list_latter_half = collections.deque()

    max_n_h = int(np.floor(h_map_size / 2.0))
    max_n_w = int(np.floor(w_map_size / 2.0))
    for h_index in tqdm.tqdm(range(-max_n_h, 1)):  # do not need to run until max_n_h+1 because of symetry.
        for w_index in range(-max_n_w, max_n_w + 1):
            # generate dataset with Fourier basis noise
            fourier_noise = AddFourierNoise(h_index, w_index, eps, norm_type=norm_type)
            dataset = dataset_builder(train=False, normalize=True, optional_transform=[fourier_noise])
            if num_samples != -1:
                num_samples = min(num_samples, len(dataset))
                indices = [i for i in range(num_samples)]
                dataset = torch.utils.data.Subset(dataset, indices)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

            with torch.autograd.no_grad():
                num_correct = 0.0
                for i, (x, t) in enumerate(loader):
                    model.eval()
                    x = x.to('cuda', non_blocking=True)
                    t = t.to('cuda', non_blocking=True)

                    model.zero_grad()
                    logit = model(x)
                    num_correct += get_num_correct(logit, t, topk=top_k)

                    if i == 0:
                        images_list_former_half.append(x[10])
                        if h_index != 0:
                            images_list_latter_half.appendleft(x[10])

                acc = num_correct / float(len(dataset))

                h_matrix_index = int(np.floor(h_map_size / 2)) + h_index
                w_matrix_index = int(np.floor(w_map_size / 2)) + w_index
                error_matrix[h_matrix_index, w_matrix_index] = 1.0 - acc

                if h_index != 0:
                    error_matrix[h_map_size - h_matrix_index - 1,
                                 w_map_size - w_matrix_index - 1] = 1.0 - acc

            # print('({h_index},{w_index}) error: {error}'.format(h_index=h_index, w_index=w_index, error=1.0-acc))
        if orator:
            print(error_matrix)

    # logging
    torch.save(error_matrix, os.path.join(log_dir, 'fhmap_data' + suffix + '.pth'))
    images_list_former_half.extend(images_list_latter_half)
    images_list = list(images_list_former_half)
    torchvision.utils.save_image(torch.stack(images_list, dim=0), os.path.join(log_dir, 'example_images' + suffix + '.png'), nrow=w_map_size)
    sns.heatmap(error_matrix.numpy(), vmin=0.0, vmax=1.0, cmap="jet", cbar=True, xticklabels=False, yticklabels=False)
    plt.savefig(os.path.join(log_dir, 'fhmap' + suffix + '.png'))
    plt.close('all')  # this is needed for continuous figure generation.


if __name__ == '__main__':
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=False, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)

    x_list = []
    for index_h in tqdm.tqdm(range(-16, 16)):
        for index_w in range(-16, 16):

            for i, (x, t) in enumerate(loader):
                x = AddFourierNoise(index_h, index_w, 16.0)(x)
                x_list.append(x[1, :, :, :])
                break

    os.makedirs('../logs', exist_ok=True)
    torchvision.utils.save_image(x_list, '../logs/fourer_heatmap_test.png', nrow=32)
