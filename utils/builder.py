import os

import torch
import torch.nn as nn
import yaml

import datasets
import models
import utils


def load_cfg(file: str):
    with open(file) as f:
        cfg = yaml.safe_load(f)

    cfg['dataset']['image_size'] = tuple([int(i) for i in cfg['dataset']['image_size'].split('x')])
    return cfg


class Builder:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def build_dataset(self):
        cfg_dataset = self.cfg['dataset']

        if cfg_dataset['name'] == 'Cityscapes':
            dataset = datasets.cityscapes.Cityscapes(self.cfg)
        else:
            raise NotImplementedError('Wrong dataset name.')
        return dataset

    def build_model(self, pretrained=False) -> torch.nn.Module:
        cfg_model = self.cfg['model']

        if cfg_model['name'] == 'UNet':
            model = models.unet.UNet(cfg_model['num_classes'])
        elif cfg_model['name'] == 'AR_UNet':
            model = models.ar_unet.AR_UNet(cfg_model['num_classes'])
        elif cfg_model['name'] == 'DeepLabV3plus':
            model = models.deeplabv3plus.DeepLabV3plus('xception', 16, cfg_model['num_classes'])
        else:
            raise NotImplementedError('Wrong model name.')

        if pretrained:
            pretrained_weights_path = self.cfg[cfg_model['name']]['pretrained_weights']
            if os.path.exists(pretrained_weights_path):
                model.load_state_dict(torch.load(pretrained_weights_path))
            else:
                print('FileNotFound: pretrained_weights ({})'.format(cfg_model['name']))
        return model

    def build_criterion(self) -> nn.Module:
        cfg_criterion: dict = self.cfg[self.cfg['model']['name']]['criterion']

        if cfg_criterion['name'] == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss(ignore_index=self.cfg['dataset']['ignore_index'])
        else:
            raise NotImplementedError('Wrong criterion name.')
        return criterion

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        cfg_optim: dict = self.cfg[self.cfg['model']['name']]['optimizer']

        if cfg_optim['name'] == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg_optim['lr'],
                                        momentum=cfg_optim['momentum'], weight_decay=cfg_optim['weight_decay'])
        elif cfg_optim['name'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg_optim['lr'], weight_decay=cfg_optim['weight_decay'])
        else:
            raise NotImplementedError('Wrong optimizer name.')
        return optimizer

    def build_scheduler(self, optimizer: torch.optim.Optimizer):
        cfg_scheduler: dict = self.cfg[self.cfg['model']['name']]['scheduler']

        if cfg_scheduler['name'] == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg_scheduler['factor'],
                                                                   patience=cfg_scheduler['patience'],
                                                                   min_lr=cfg_scheduler['min_lr'])
        elif cfg_scheduler['name'] == 'WarmupPolyLR':
            scheduler = utils.lr_scheduler.WarmupPolyLR(optimizer, self.cfg[self.cfg['model']['name']]['iter'])
        else:
            raise NotImplementedError('Wrong scheduler name.')
        return scheduler
