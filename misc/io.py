import os
import sys

import torch

def print_metric_dict(epoch, num_epochs, metric_dict:dict, reverse:bool=True, overwrite:bool=False, mode:str='train'):
    """
    print metric dict info to command line
    Args:
        epoch:          current epoch
        num_epochs:     total number of training epochs
        metric_dict:    metric_dict
        reverse:        print reverse order
        overwrite:      overwrite print message or not
    """
    modes = set(['train', 'val', 'test'])
    if mode not in modes: raise ValueError('mode is invalid')

    print_message =  '\r\033[K' if overwrite is True else ''
    # add epoch
    if (mode=='train') or (mode=='val'): 
        print_message += 'epoch [{:d}/{:d}] '.format(epoch+1, num_epochs)
    # add mode
    print_message += ' ({}) '.format(mode)
    # add metric
    dict_items = reversed(list(metric_dict.items())) if reverse is True else metric_dict.items()
    for k,v in dict_items:
        print_message += '{}:{:.4f} '.format(k,v) 
    # add new line
    print_message += '' if overwrite is True else '\n'
    sys.stdout.write(print_message)
    sys.stdout.flush()


def load_model(model, path, orator=True):
    model.load_state_dict(torch.load(path))
    if orator: print('>>> Model was loaded from "{}"'.format(path))

def save_model(model, path, orator=True):
    torch.save(
        model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
        path
    )
    if orator: print('>>> Model was saved to "{}"'.format(path))

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']

def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'model': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, path)