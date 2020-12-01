import configparser
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import tqdm

import utils


def make_plt_subplot(nrows: int, ncols: int, index: int, title: str, image):
    plt.subplot(nrows, ncols, index)
    plt.title(title)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])


# 데이터셋 불러오는 코드 검증
def show_dataset(images: torch.Tensor, masks: torch.Tensor):
    to_pil_image = torchvision.transforms.ToPILImage()

    for i in range(images.shape[0]):
        make_plt_subplot(1, 2, 1, 'Input image', to_pil_image(images[i].squeeze().cpu()))
        make_plt_subplot(1, 2, 2, 'Groundtruth', to_pil_image(masks[i].squeeze().cpu()))
        plt.show()


# 가중치 초기화
def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.normal_(m.bias)
    elif type(m) == nn.BatchNorm2d:
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# 가중치 초기화 (제안하는 방법)
def init_weights_proposed(m):
    if type(m) == nn.Conv2d:
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    elif type(m) == nn.BatchNorm2d:
        nn.init.zeros_(m.weight)


# 설정 불러오기
def load_config(ini_file: str):
    section = ini_file.split('/')[-1].split('.')[0]
    parser = configparser.ConfigParser()
    parser.read(ini_file, encoding='utf-8')
    config = {
        'batch_size': parser.getint(section, 'batch_size'),
        'epoch': parser.getint(section, 'epoch'),
        'image_size': parser.getint(section, 'image_size'),
        'lr': parser.getfloat(section, 'lr'),
        'num_classes': parser.getint(section, 'num_classes'),
        'num_workers': parser.getint(section, 'num_workers'),
        'pretrained_weights': parser[section]['pretrained_weights'],
    }
    return config, section


# 데이터셋 설정
def init_dataset(config: dict):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(config['image_size']),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    target_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(config['image_size']),
        torchvision.transforms.ToTensor(),
    ])
    trainset = utils.dataset.Cityscapes(root='../../data/cityscapes',
                                        split='train',
                                        mode='fine',
                                        target_type='semantic',
                                        transform=transform,
                                        target_transform=target_transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=config['batch_size'],
                                              shuffle=True,
                                              num_workers=config['num_workers'],
                                              pin_memory=True)
    testset = utils.dataset.Cityscapes(root='../../data/cityscapes',
                                       split='val',
                                       mode='fine',
                                       target_type='semantic',
                                       transform=transform,
                                       target_transform=target_transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=config['batch_size'],
                                             shuffle=False,
                                             num_workers=config['num_workers'],
                                             pin_memory=True)

    return trainset, trainloader, testset, testloader


def save_groundtruth(testset):
    testloader = torch.utils.data.DataLoader(testset)

    image_names = []
    for image_path in testset.images:
        image_name = image_path.replace('\\', '/').split('/')[-1]
        image_names.append(image_name)

    result_dir = os.path.join('demo', 'groundtruth')
    os.makedirs(result_dir, exist_ok=True)
    for i, (_, mask) in enumerate(tqdm.tqdm(testloader, desc='GroundTruth')):
        plt.imsave(os.path.join(result_dir, image_names[i]), mask.squeeze())
