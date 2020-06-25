import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import random
import numpy as np
import torch
import torchvision

from fhmap.fourier_base import generate_fourier_base


class AddFourierNoise(object):
    def __init__(self, h_index: int, w_index: int, eps: float):
        """
        Args
            h_index: index of fourier basis about hight direction 
            w_index: index of fourier basis about width direction 
            eps: size of noise(perturbation)
        """
        assert eps >= 0.0
        self.h_index = h_index
        self.w_index = w_index
        self.eps = eps

    def __call__(self, x: torch.tensor):
        c, h, w = x.shape[-3:]
        assert c == 3
        assert h >= 1 or w >= 1
        assert abs(self.h_index) <= np.floor(h / 2) and abs(self.w_index) <= np.floor(w / 2)

        fourier_base = generate_fourier_base(h, w, self.h_index, self.w_index)  # normalized fourier base
        fourier_base /= fourier_base.norm()

        eps_l2 = np.sqrt(((self.eps / 255.0)**2.0) * h * w)
        fourier_base *= eps_l2

        fourier_noise = fourier_base.unsqueeze(0).repeat(c, 1, 1)
        # fourier_noise /= fourier_noise.norm()

        fourier_noise[0, :, :] *= random.randrange(-1, 2, 2)
        fourier_noise[1, :, :] *= random.randrange(-1, 2, 2)
        fourier_noise[2, :, :] *= random.randrange(-1, 2, 2)

        return torch.clamp(x + fourier_noise, min=0.0, max=1.0)


if __name__ == '__main__':
    import tqdm

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
