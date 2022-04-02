import logging
import os
import random
import sys

import numpy as np
import torch
from torch.backends import cudnn

from config_yaml import load_config
from networks.net_factory_3d import net_factory_3d

def train(config):
    # create the model
    model_config = config['model']
    model = net_factory_3d(net_type=model_config['name'],
                           in_chns=model_config['in_channels'],
                           class_num=model_config['class_num'])
    # create loss criterion
    # create evaluation metric
    # create data loaders
    # create the optimizer
    # create learning rate adjustment strategy




if __name__ == '__main__':
    # load configuration
    config = load_config()
    # snapshot_path
    snapshot_path = "../model/{}/{}".format(config.basic.exp, config.model.name)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config['snapshot_path'] = snapshot_path
    # log experiment configuration
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(config))
    basic_config = config['basic']
    # seed config
    seed = basic_config.pop('seed', None)
    if seed is not None:
        logging.info(f'Seed the RNG for all devices with {seed}')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    deterministic = basic_config.pop('deterministic', False)
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        logging.warning('Using CuDNN deterministic setting. This may slow down the training!')
        cudnn.benchmark = False
        cudnn.deterministic = True
    # start training
    train(config)







