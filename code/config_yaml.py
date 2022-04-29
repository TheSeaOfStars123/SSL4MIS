import argparse

import torch
import yaml


def load_config():
    parser = argparse.ArgumentParser(description='UNet3D')
    parser.add_argument('--config', type=str, help='path to the YAML config file', required=True)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))
    config['basic']['config'] = args.config
    # get a devide to train on
    device_str = config.get('device', None)
    if device_str is not None:
        if device_str.startswith('cuda') and not torch.cuda.is_available():
            device_str = 'cpu'
    else:
        device_str = "cuda:0" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    config['device'] = device
    return config

