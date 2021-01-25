import os

import matplotlib.colors
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data
import tqdm

import utils

if __name__ == '__main__':
    # 0. Load config
    config = utils.load_config()
    print('Activated model: {}'.format(config['model']))

    # 1. Dataset
    dataset = utils.Cityscapes(config)
    _, _, testset, testloader = dataset.set_cityscapes()

    # 2. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = utils.get_model(config, pretrained=True).to(device)
    model.eval()

    # 이미지 이름 불러오기
    image_names = []
    for image_path in testset.images:
        image_name = image_path.replace('\\', '/').split('/')[-1]
        image_names.append(image_name)

    # label colormap 설정
    cmap = matplotlib.colors.ListedColormap(dataset.get_cityscapes_colormap(short=True))

    # 예측 결과 저장
    step = 0
    result_dir = os.path.join('demo', config['model'])
    groundtruth_dir = os.path.join('demo', 'groundtruth')
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(groundtruth_dir, exist_ok=True)
    for images, masks in tqdm.tqdm(testloader, desc='Demo'):
        # mask에 255를 곱하여 0~1 사이의 값을 0~255 값으로 변경 + 채널 차원 제거
        masks.mul_(255).squeeze_(dim=1)

        images, masks = images.to(device), masks.type(torch.LongTensor)

        # 예측
        with torch.no_grad():
            masks_pred = model(images)
            masks_pred = F.log_softmax(masks_pred, dim=1)
            masks_pred = torch.argmax(masks_pred, dim=1)

        # 1 배치단위 처리
        assert masks.shape[0] == masks_pred.shape[0]
        for i in range(masks.shape[0]):
            plt.imsave(os.path.join(result_dir, image_names[step]), masks_pred[i].cpu(), cmap=cmap)
            plt.imsave(os.path.join(groundtruth_dir, image_names[step]), masks[i], cmap=cmap)
            step += 1
