import os

import torch
import torch.nn as nn
import yaml

import datasets
import models


def load_cfg() -> dict:
    with open('cfgs/main.yaml') as f:
        main = yaml.safe_load(f)

    with open(main['cfg']) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg['model']['name'] = main['model']
    return cfg


class Builder:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def build_dataset(self, split: str):
        cfg_dataset = self.cfg['dataset']

        if cfg_dataset['name'] == 'Cityscapes':
            dataset_impl = datasets.cityscapes.Cityscapes(self.cfg)
            dataset, dataloader = dataset_impl.get_dataloader(split)
        else:
            raise NotImplementedError('Wrong dataset name.')
        return dataset_impl, dataset, dataloader

    def build_model(self, num_classes: int, pretrained=False) -> torch.nn.Module:
        cfg_model_name = self.cfg['model']['name']

        if cfg_model_name == 'UNet':
            model = models.unet.UNet(num_classes)
        elif cfg_model_name == 'AR_UNet':
            model = models.ar_unet.AR_UNet(num_classes)
        elif cfg_model_name == 'DeepLabV3plus':
            model = models.deeplabv3plus.DeepLabV3plus(self.cfg[cfg_model_name]['backbone'], 16, num_classes)
        else:
            raise NotImplementedError('Wrong model name.')

        if pretrained:
            pretrained_weights_path = self.cfg[cfg_model_name]['pretrained_weights']
            if os.path.exists(pretrained_weights_path):
                model.load_state_dict(torch.load(pretrained_weights_path))
            else:
                print(f'FileNotFound: pretrained_weights ({cfg_model_name})')
        return model

    def build_criterion(self, ignore_index: int) -> nn.Module:
        cfg_criterion = self.cfg[self.cfg['model']['name']]['criterion']

        if cfg_criterion['name'] == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            raise NotImplementedError('Wrong criterion name.')
        return criterion

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        cfg_optim = self.cfg[self.cfg['model']['name']]['optimizer']

        if cfg_optim['name'] == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg_optim['lr'],
                                        momentum=cfg_optim['momentum'], weight_decay=cfg_optim['weight_decay'])
        elif cfg_optim['name'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg_optim['lr'], weight_decay=cfg_optim['weight_decay'])
        elif cfg_optim['name'] == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_optim['lr'])
        else:
            raise NotImplementedError('Wrong optimizer name.')
        return optimizer

    def build_scheduler(self, optimizer: torch.optim.Optimizer):
        cfg_scheduler = self.cfg[self.cfg['model']['name']]['scheduler']

        if cfg_scheduler['name'] == 'CosineAnnealingWarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg_scheduler['T_0'])
        else:
            raise NotImplementedError('Wrong scheduler name.')
        return scheduler
