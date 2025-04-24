import argparse
import os
import yaml
import torch
import datasets
import utils
from models import DenoisingDiffusion, DiffusiveRestoration

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='NAR-DIFF')
    parser.add_argument("--config", default='config.yml', type=str,help="Path to the YAML configuration")
    parser.add_argument('--mode', type=str, default='evaluation', help='training/evaluation')
    parser.add_argument('--resume', default='ckpt/', type=str,help='pre-trained model checkpoint')
    parser.add_argument("--image_folder", default='results/', type=str,help="Output directory")
    args = parser.parse_args()
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def main():
    args, config = parse_args_and_config()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    print("=> using dataset '{}'".format(config.data.val_dataset))
    DATASET = datasets.__dict__[config.data.type](config)
    _, val_loader = DATASET.get_loaders()
    print("=> creating denoising-diffusion model")
    diffusion = DenoisingDiffusion(args, config)
    model = DiffusiveRestoration(diffusion, args, config)
    model.restore(val_loader)


if __name__ == '__main__':
    main()