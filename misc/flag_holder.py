import os
import sys
import json
from datetime import datetime

class FlagHolder(object):
    def __init__(self):
        self._dict = None

    def __getattr__(self, key):
        return self._dict[key]
        
    def initialize(self, **kwargs):
        self._dict = {}
        self._dict['time'] = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        
        for k, v in kwargs.items():
            self._dict[k] = v

    def summary(self):
        print('===== Flag summary =====')
        for k, v in self._dict.items():
            print('{k}: {v}'.format(k=k, v=v))
        print('=== End flag summary ===')

    def dump(self, path):
        """
        dump all flag to json file.
        """
        # check extension
        base_path, ext = os.path.splitext(path)
        if ext == '.json':
            pass
        else:
            path = base_path + '.json' 

        with open(path, mode='w') as f:
            json.dump(self._dict, f, indent=2)

if __name__ == '__main__':
    flag = {
        'model': 'resnet18',
        'batch_size': 128,
        'train': True 
    }

    FLAGS = FlagHolder()
    FLAGS.initialize(**flag)
    FLAGS.summary()
    FLAGS.dump('../logs/flags.json')