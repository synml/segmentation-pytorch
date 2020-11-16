import configparser
import os

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data
import torchvision
import tqdm

import model.unet
import model.proposed
import utils.dataset

ini_file = 'model/unet.ini'
section = ini_file.split('/')[-1].split('.')[0]
parser = configparser.ConfigParser()
parser.read(ini_file, encoding='utf-8')
config = {
    'batch_size': parser.getint(section, 'batch_size'),
    'image_size': parser.getint(section, 'image_size'),
    'num_classes': parser.getint(section, 'num_classes'),
    'num_workers': parser.getint(section, 'num_workers'),
    'pretrained_weights': parser[section]['pretrained_weights'],
    'result_dir': 'result/'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터셋 설정
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(config['image_size']),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
target_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(config['image_size']),
    torchvision.transforms.ToTensor(),
])
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

# 모델 설정
if section == 'unet':
    model = model.unet.UNet(3, config['num_classes']).to(device)
elif section == 'proposed':
    model = model.proposed.Proposed(3, config['num_classes']).to(device)
model.load_state_dict(torch.load(config['pretrained_weights']))

# 이미지 이름 저장
image_names = []
for image_path in testset.images:
    image_name = image_path.replace('\\', '/').split('/')[-1]
    image_names.append(image_name)

# 예측 결과 저장
step = 0
os.makedirs(config['result_dir'], exist_ok=True)
for images, _ in tqdm.tqdm(testloader, desc='Demo'):
    # 이미지와 정답 정보를 GPU로 복사
    images = images.to(device)

    # 예측
    with torch.no_grad():
        masks_pred = model(images)
        masks_pred = F.softmax(masks_pred, dim=1)
        masks_pred = torch.argmax(masks_pred, dim=1, keepdim=True)

    # 배치 단위의 mask를 1개씩 분해
    for mask in masks_pred:
        plt.imsave(os.path.join(config['result_dir'], image_names[step]), mask.cpu().squeeze())
        step += 1
