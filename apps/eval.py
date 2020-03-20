import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import click
import tqdm
import random

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns

from misc.flag_holder import FlagHolder
from misc.io import load_model
from misc.metric import get_num_correct
from misc.data import DatasetBuilder
from misc.model import ModelBuilder
from fhmap.fourier_heatmap import AddFourierNoise

# options
@click.command()
# model
@click.option('-a', '--arch', type=str, required=True)
# model
@click.option('-w', '--weight', type=str, required=True, help='model weight path')
# data
@click.option('-d', '--dataset', type=str, required=True)
@click.option('--dataroot', type=str, default='../data', help='path to dataset root')
@click.option('-j', '--num_workers', type=int, default=8)
@click.option('-N', '--batch_size', type=int, default=1024)
@click.option('--num_samples', type=int, default=1024)
# fourier heatmap
@click.option('--h_map_size', type=int, default=32)
@click.option('--w_map_size', type=int, default=32)
@click.option('--eps', type=float, default=32)
@click.option('-k', '--top_k', type=int, default=1)
# log
@click.option('-l', '--log_dir', type=str, required=True)
@click.option('-s', '--suffix', type=str, default='')


def main(**kwargs):
    eval(**kwargs)

def eval(**kwargs):
    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)
    FLAGS.summary()

    # dataset
    dataset_builder = DatasetBuilder(name=FLAGS.dataset, root_path=FLAGS.dataroot)

    # model (load from checkpoint) 
    num_classes = dataset_builder.num_classes
    model = ModelBuilder(num_classes=num_classes, pretrained=False)[FLAGS.arch].cuda()
    load_model(model, FLAGS.weight)
    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)

    # Fourier Heatmap
    error_matrix = torch.zeros(FLAGS.h_map_size, FLAGS.w_map_size).float() 
    images_list  = []

    for h_index in tqdm.tqdm(range(-int(np.floor(FLAGS.h_map_size/2)), FLAGS.h_map_size-int(np.floor(FLAGS.h_map_size/2)))):
        for w_index in range(-int(np.floor(FLAGS.w_map_size/2)), FLAGS.w_map_size-int(np.floor(FLAGS.w_map_size/2))):

            # generate dataset with Fourier basis noise
            fourier_noise = AddFourierNoise(h_index, w_index, FLAGS.eps)
            dataset = dataset_builder(train=False, normalize=True, optional_transform=[fourier_noise])
            if FLAGS.num_samples != -1:
                num_samples = min(FLAGS.num_samples, len(dataset))
                indices = [i for i in range(num_samples)]
                dataset = torch.utils.data.Subset(dataset, indices)
            loader = torch.utils.data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.num_workers, pin_memory=True)

            with torch.autograd.no_grad():
                num_correct = 0.0
                for i, (x,t) in enumerate(loader):
                    model.eval()
                    x = x.to('cuda', non_blocking=True)
                    t = t.to('cuda', non_blocking=True)

                    model.zero_grad()
                    logit = model(x)
                    num_correct += get_num_correct(logit, t, topk=FLAGS.top_k)

                    if i==0:
                        images_list.append(x[10])
                
                acc = num_correct / float(len(dataset))
                error_matrix[h_index+int(np.floor(FLAGS.h_map_size/2)), w_index+int(np.floor(FLAGS.w_map_size/2))] = 1.0 - acc

            #print('({h_index},{w_index}) error: {error}'.format(h_index=h_index, w_index=w_index, error=1.0-acc))
        print(error_matrix)

    # logging
    os.makedirs(FLAGS.log_dir, exist_ok=True)
    torch.save(error_matrix, os.path.join(FLAGS.log_dir, 'fhmap_data'+FLAGS.suffix+'.pth'))
    torchvision.utils.save_image(torch.stack(images_list, dim=0), os.path.join(FLAGS.log_dir, 'example_images'+FLAGS.suffix+'.png'), nrow=FLAGS.w_map_size)
    sns.heatmap(error_matrix.numpy(), vmin=0.0, vmax=1.0, cmap="jet", cbar=True, xticklabels=False, yticklabels=False)
    plt.savefig(os.path.join(FLAGS.log_dir, 'fhmap'+FLAGS.suffix+'.png'))
    

if __name__ == '__main__':
    main()