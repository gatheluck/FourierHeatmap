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
import hydra
import omegaconf

from misc.metric import get_num_correct
from fhmap.fourier_base import generate_fourier_base
from fhmap.fourier_heatmap import AddFourierNoise
from fhmap.fourier_basis_augmented_dataset import FourierBasisAugmentedDataset


def create_fourier_heatmap_fba_dataset(model, dataset_builder, h_map_size: int, w_map_size: int, eps: float, norm_type: str, num_samples: int, batch_size: int, num_workers: int, log_dir: str, top_k: int = 1, suffix: str = '', shuffle: bool = False, orator: bool = False, **kwargs):
    """
    Create Fourier heatmap by FourierBasisAugmentedDataset.
    Generally this function is slower than 'fourier_heatmap.create_fourier_heatmap'.

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

    # base dataset
    dataset = dataset_builder(train=False, normalize=False, optional_transform=[])
    if num_samples != -1:
        num_samples = min(num_samples, len(dataset))
        indices = [i for i in range(num_samples)]
        dataset = torch.utils.data.Subset(dataset, indices)

    # Fourier Heatmap
    error_matrix = torch.zeros(h_map_size, w_map_size).float()

    max_n_h = int(np.floor(h_map_size / 2.0))
    max_n_w = int(np.floor(w_map_size / 2.0))

    # loop over half of Fourier space
    with tqdm.tqdm(total=np.ceil((h_map_size * w_map_size) / 2), ncols=80) as pbar:
        for h_index in range(-max_n_h, 1):  # do not need to run until max_n_h+1 because of symetry.
            # if h_index == 0, only loop negative side
            range_w_index = range(-max_n_w, max_n_w + 1) if h_index != 0 else range(-max_n_w, 1)
            for w_index in range_w_index:

                # prepare dataset and loader
                fba_dataset = FourierBasisAugmentedDataset(dataset,
                                                           input_size=dataset_builder.input_size,
                                                           mean=dataset_builder.mean,
                                                           std=dataset_builder.std,
                                                           h_index=h_index,
                                                           w_index=w_index,
                                                           eps=eps,
                                                           randomize_index=False,
                                                           normalize=True,
                                                           mode='index')
                loader = torch.utils.data.DataLoader(fba_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

                # calc accuracy
                with torch.autograd.no_grad():
                    num_correct = 0.0
                    for i, (x, t, _) in enumerate(loader):
                        model.eval()
                        x = x.to('cuda', non_blocking=True)
                        t = t.to('cuda', non_blocking=True)

                        model.zero_grad()
                        logit = model(x)
                        num_correct += get_num_correct(logit, t, topk=top_k)

                        # if i == 0:
                        #     torchvision.utils.save_image(x[:16, :, :, :], './sample_img_h{h}_w{w}.png'.format(h=h_index, w=w_index))

                    acc = num_correct / float(len(fba_dataset))

                    h_matrix_index = int(np.floor(h_map_size / 2)) + h_index
                    w_matrix_index = int(np.floor(w_map_size / 2)) + w_index

                    # add to error matrix
                    error_matrix[h_matrix_index, w_matrix_index] = 1.0 - acc
                    error_matrix[h_map_size - h_matrix_index - 1, w_map_size - w_matrix_index - 1] = 1.0 - acc

                    pbar.set_postfix(collections.OrderedDict(h_index=h_index, w_index=w_index, err=1.0 - acc))
                    pbar.update()

                del loader
                del fba_dataset

    # logging
    torch.save(error_matrix, os.path.join(log_dir, 'fhmap_data' + suffix + '.pth'))
    sns.heatmap(error_matrix.numpy(), vmin=0.0, vmax=1.0, cmap="jet", cbar=True, xticklabels=False, yticklabels=False)
    plt.savefig(os.path.join(log_dir, 'fhmap' + suffix + '.png'))
    plt.close('all')  # this is needed for continuous figure generation.


@hydra.main(config_path='../conf/config.yaml')
def test(cfg: omegaconf.DictConfig):
    def load_model(model, path):
        if not os.path.exists(path):
            raise FileNotFoundError('path "{path}" does not exist.'.format(path=path))

        print('loading model weight from {path}'.format(path=path))

        # load weight from .pth file.
        if path.endswith('.pth'):
            statedict = collections.OrderedDict()
            for k, v in torch.load(path).items():
                if k.startswith('model.'):
                    k = '.'.join(k.split('.')[1:])

                statedict[k] = v

            # model.load_state_dict(torch.load(path))
            model.load_state_dict(statedict)
        # load weight from checkpoint.
        elif path.endswith('.ckpt'):
            checkpoint = torch.load(path)
            if 'state_dict' in checkpoint.keys():
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                raise ValueError('this checkponint do not inculdes state_dict')
        else:
            raise ValueError('path is not supported type of extension.')

    from misc.data import DatasetBuilder
    from misc.model import ModelBuilder
    dataset_builder = DatasetBuilder(root_path=os.path.join(hydra.utils.get_original_cwd(), '../data'), **cfg.dataset)
    model_builder = ModelBuilder(10, pretrained=False)
    model = model_builder['resnet56'].cuda()

    if cfg.filepath is not None:
        load_model(model, cfg.filepath)

    create_fourier_heatmap_fba_dataset(model,
                                       dataset_builder,
                                       h_map_size=31,
                                       w_map_size=31,
                                       eps=4.0,
                                       norm_type='l2',
                                       num_samples=1024,
                                       batch_size=1024,
                                       num_workers=8,
                                       log_dir='.')


if __name__ == '__main__':
    test()
