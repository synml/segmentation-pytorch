import os

import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
import torch.utils.data
import tqdm

import utils

if __name__ == '__main__':
    # 0. Load cfg and create components builder
    cfg = utils.builder.load_cfg()
    builder = utils.builder.Builder(cfg)

    # 1. Dataset
    valset, valloader = builder.build_dataset('val')

    # 2. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = builder.build_model(valset.num_classes, pretrained=True).to(device)
    model_name = cfg['model']['name']
    amp_enabled = cfg['model']['amp_enabled']
    print(f'Activated model: {model_name}')

    # 이미지 이름 불러오기
    image_names = []
    for image_path in valset.images:
        image_name = image_path.replace('\\', '/').split('/')[-1]
        image_names.append(image_name)

    # label colormap 설정
    cmap = matplotlib.colors.ListedColormap(np.divide(valset.get_colormap(), 255).tolist())

    # 예측 결과 저장
    step = 0
    result_dir = os.path.join('demo', model_name.lower())
    groundtruth_dir = os.path.join('demo', 'groundtruth')
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(groundtruth_dir, exist_ok=True)
    for images, targets in tqdm.tqdm(valloader, desc='Demo'):
        images = images.to(device)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            with torch.no_grad():
                outputs = model(images)
                outputs = torch.argmax(outputs, dim=1)

        # 1 배치단위 처리
        assert targets.shape[0] == outputs.shape[0]
        for i in range(targets.shape[0]):
            plt.imsave(os.path.join(result_dir, image_names[step]), outputs[i].cpu(), cmap=cmap, vmin=0, vmax=cmap.N)
            plt.imsave(os.path.join(groundtruth_dir, image_names[step]), targets[i], cmap=cmap, vmin=0, vmax=cmap.N)
            step += 1
