import os

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data
import tqdm

import utils.utils


if __name__ == '__main__':
    # 0. Load config
    model_name, config = utils.utils.load_config()
    print('Activated model: {}'.format(model_name))

    # 1. Dataset
    _, _, testset, testloader = utils.utils.init_cityscapes_dataset(config)

    # 2. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = utils.utils.get_model(model_name, 3, config['num_classes'], config['pretrained_weights']).to(device)

    # 이미지 이름 불러오기
    image_names = []
    for image_path in testset.images:
        image_name = image_path.replace('\\', '/').split('/')[-1]
        image_names.append(image_name)

    # 예측 결과 저장
    step = 0
    result_dir = os.path.join('demo', model_name.lower())
    groundtruth_dir = os.path.join('demo', 'groundtruth')
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(groundtruth_dir, exist_ok=True)
    for images, masks in tqdm.tqdm(testloader, desc='Demo'):
        images = images.to(device)

        # 예측
        with torch.no_grad():
            masks_pred = model(images)
            masks_pred = F.log_softmax(masks_pred, dim=1)
            masks_pred = torch.argmax(masks_pred, dim=1)

        # 1 배치단위 처리
        assert masks.shape[0] == masks_pred.shape[0]
        for i in range(masks.shape[0]):
            plt.imsave(os.path.join(result_dir, image_names[step]), masks_pred[i].cpu())
            plt.imsave(os.path.join(groundtruth_dir, image_names[step]), masks[i].squeeze())
            step += 1
