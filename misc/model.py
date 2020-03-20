import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base)

from collections import namedtuple

import torch
import torchvision

from misc.models.alexnet import alexnet_v2
from misc.models.wideresnet import wideresnet16, wideresnet28, wideresnet40
from misc.models.cifar10_resnet import resnet56

class ModelBuilder(object):
    MODEL_CONFIG = {
        'alexnet':      [''],
        'vgg':          '16 16_bn 19 19_bn'.split(),
        'resnet':       '18 34 50 56 101 152'.split(),
        'wideresnet':   '16 28 40'.split(),
    }

    def __init__(self, num_classes:int, pretrained=False, inplace=True, use_bn=True):
        assert num_classes > 0

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.inplace = inplace
        self.use_bn = use_bn

    def __getitem__(self, name):
        assert name in self.available_models
        return self._get_classifier(name, num_classes=self.num_classes, pretrained=self.pretrained, inplace=self.inplace, use_bn=self.use_bn)

    @property
    def available_models(self):
        return [arch+depth for arch  in self.MODEL_CONFIG.keys()
                           for depth in self.MODEL_CONFIG[arch]]

    def _get_classifier(self, name, num_classes=1000, pretrained=False, inplace=True, use_bn=True):
        # get pretrained model
        if pretrained:
            # call torchvision function
            func = eval('torchvision.models.{name}'.format(name=name))
            model = func(pretrained=True)
        
            # replace fc
            if num_classes != 1000:
                self._replace_final_fc(name, model, num_classes)

        # get not pretrained model
        else:
            if name == 'alexnet':
                model = alexnet_v2(num_classes=num_classes, use_bn=use_bn)
            # call our function
            elif name.startswith('wideresnet'):
                func = eval('{name}'.format(name=name))
                model = func(num_classes=num_classes)
            elif name in ['resnet56']:
                assert num_classes==10, "this resnet is for CIFAR"
                func = eval('{name}'.format(name=name))
                model = func()
            # call torchvision function
            else:
                func = eval('torchvision.models.{name}'.format(name=name))
                model = func(pretrained=False, num_classes=num_classes)

        # relpace ReLU.inplace 
        for m in model.modules():
            if isinstance(m, torch.nn.ReLU):
                m.inplace = inplace

        return model

    def _replace_final_fc(name, model, num_classes):
        if name == 'alexnet' or name.startswith('vgg'):
            num_features = model.classifier[6].in_features
            model.classifier[6] = torch.nn.Linear(num_features, num_classes)
        elif name.startswith('resnet'):
            num_features = model.fc.in_features
            model.fc = torch.nn.Linear(num_features, num_classes)
        else:
            raise NotImplementedError

if __name__ == '__main__':
    model_builder = ModelBuilder(10, pretrained=False)
    print(model_builder.available_models)
    model = model_builder['resnet50']
    print(model)