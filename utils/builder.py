import os
import platform

import torch
import torch.distributed.optim
import torch.nn as nn
import torch.utils.data
import yaml

import datasets
import models
import utils


def load_cfg() -> dict:
    with open('cfgs/main.yaml') as f:
        main = yaml.safe_load(f)

    with open(main['cfg']) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg['model'] = {}
    cfg['model']['name'] = main['model']
    cfg['model']['amp_enabled'] = main['amp_enabled']
    cfg['resume_training'] = main['resume_training']
    cfg['fine_tuning_batchnorm'] = main['fine_tuning_batchnorm']
    cfg['ddp_enabled'] = main['ddp_enabled']
    return cfg


class Builder:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def build_dataset(self, dataset_type: str):
        cfg_dataset = self.cfg['dataset']
        root = cfg_dataset['root']
        batch_size = self.cfg[self.cfg['model']['name']]['batch_size']
        pin_memory = cfg_dataset['pin_memory']
        if platform.system() == 'Windows':
            num_workers = 0
        else:
            num_workers = self.cfg['dataset']['num_workers']
        if dataset_type == 'train':
            transforms = datasets.transforms.Transforms(self.cfg, augmentation=True)
            shuffle = True
        else:
            transforms = datasets.transforms.Transforms(self.cfg)
            shuffle = False

        # Dataset
        if cfg_dataset['name'] == 'Cityscapes':
            dataset = datasets.cityscapes.Cityscapes(root, dataset_type, mode='fine', target_type='semantic',
                                                     transforms=transforms)
        else:
            raise NotImplementedError('Wrong dataset name.')

        # Dataloader
        if self.cfg['ddp_enabled']:
            sampler = torch.utils.data.DistributedSampler(dataset)
            shuffle = False
        else:
            sampler = None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                                                 num_workers=num_workers, pin_memory=pin_memory)
        return dataset, dataloader

    def build_model(self, num_classes: int, pretrained=False) -> torch.nn.Module:
        cfg_model_name = self.cfg['model']['name']

        if cfg_model_name == 'UNet':
            model = models.unet.UNet(num_classes)
        elif cfg_model_name == 'EAR_Net':
            model = models.ear_net.EAR_Net(num_classes)
        elif cfg_model_name == 'DeepLabV3plus':
            model = models.deeplabv3plus.DeepLabV3plus(self.cfg[cfg_model_name]['backbone'],
                                                       self.cfg[cfg_model_name]['output_stride'], num_classes)
        elif cfg_model_name == 'Proposed':
            model = models.proposed.Proposed(self.cfg[cfg_model_name]['backbone'],
                                             self.cfg[cfg_model_name]['output_stride'], num_classes)
        else:
            raise NotImplementedError('Wrong model name.')

        if pretrained:
            pretrained_weights_path = self.cfg[cfg_model_name]['pretrained_weights']
            if os.path.isfile(pretrained_weights_path):
                model.load_state_dict(torch.load(pretrained_weights_path))
            else:
                print(f'FileNotFound: pretrained_weights ({cfg_model_name})')
        return model

    def build_criterion(self, ignore_index: int) -> nn.Module:
        cfg_criterion = self.cfg[self.cfg['model']['name']]['criterion']

        if cfg_criterion['name'] == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        elif cfg_criterion['name'] == 'FocalLoss':
            criterion = utils.loss.FocalLoss(ignore_index=ignore_index,
                                             alpha=cfg_criterion['alpha'], gamma=cfg_criterion['gamma'])
        else:
            raise NotImplementedError('Wrong criterion name.')
        return criterion

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        cfg_optim = self.cfg[self.cfg['model']['name']]['optimizer']

        if self.cfg['ddp_enabled']:
            if cfg_optim['name'] == 'SGD':
                optimizer = torch.distributed.optim.ZeroRedundancyOptimizer(
                    model.parameters(), torch.optim.SGD, lr=cfg_optim['lr'], momentum=cfg_optim['momentum'],
                    weight_decay=cfg_optim['weight_decay'], nesterov=cfg_optim['nesterov']
                )
            elif cfg_optim['name'] == 'Adam':
                optimizer = torch.distributed.optim.ZeroRedundancyOptimizer(
                    model.parameters(), torch.optim.Adam, lr=cfg_optim['lr'], weight_decay=cfg_optim['weight_decay']
                )
            elif cfg_optim['name'] == 'AdamW':
                optimizer = torch.distributed.optim.ZeroRedundancyOptimizer(
                    model.parameters(), torch.optim.AdamW, lr=cfg_optim['lr']
                )
            else:
                raise NotImplementedError('Wrong optimizer name.')
        else:
            if cfg_optim['name'] == 'SGD':
                optimizer = torch.optim.SGD(
                    model.parameters(), lr=cfg_optim['lr'], momentum=cfg_optim['momentum'],
                    weight_decay=cfg_optim['weight_decay'], nesterov=cfg_optim['nesterov']
                )
            elif cfg_optim['name'] == 'Adam':
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=cfg_optim['lr'], weight_decay=cfg_optim['weight_decay']
                )
            elif cfg_optim['name'] == 'AdamW':
                optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_optim['lr'])
            else:
                raise NotImplementedError('Wrong optimizer name.')
        return optimizer

    def build_scheduler(self, optimizer: torch.optim.Optimizer):
        cfg_scheduler = self.cfg[self.cfg['model']['name']]['scheduler']

        if cfg_scheduler['name'] == 'PolyLR':
            scheduler = utils.lr_scheduler.PolyLR(optimizer, cfg_scheduler['max_iter'])
        else:
            raise NotImplementedError('Wrong scheduler name.')
        return scheduler
