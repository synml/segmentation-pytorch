import os

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data
import tqdm

import model.unet
import model.proposed
import utils.utils

# 설정 불러오기
ini_file = 'model/unet.ini'
config, section = utils.utils.load_config(ini_file)
print('{}를 불러왔습니다.'.format(ini_file.split('/')[-1]))

# 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터셋 설정
_, _, testset, testloader = utils.utils.init_voc_dataset(config)

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
result_dir = os.path.join('demo', section)
os.makedirs(result_dir, exist_ok=True)
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
        plt.imsave(os.path.join(result_dir, image_names[step]), mask.cpu().squeeze())
        step += 1
