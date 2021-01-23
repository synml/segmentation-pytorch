import os

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data

import utils


def get_feature_maps(feature_maps: dict, name: str):
    def hook(model, input, output):
        feature_maps[name] = output.detach()
    return hook


if __name__ == '__main__':
    # 0. Load config
    config = utils.load_config()
    print('Activated model: {}'.format(config['model_name']))

    # 1. Dataset
    dataset = utils.Cityscapes(config)
    _, _, testset, testloader = dataset.set_cityscapes()

    # 2. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = utils.get_model(config['model_name'], config['num_classes'], config['pretrained_weights']).to(device)

    # 이미지 가져오기
    image, _ = testset[0]
    image = image.unsqueeze(0).to(device)

    # 모델의 각 레이어에 있는 특징맵을 받아오기
    feature_maps = {}
    _ = model.encode1.register_forward_hook(get_feature_maps(feature_maps, 'encode1'))

    # 예측
    with torch.no_grad():
        mask_pred = model(image)
        mask_pred = F.log_softmax(mask_pred, dim=1)
        mask_pred = mask_pred.detach().squeeze().cpu()

    # feature map 저장
    name = 'encode1'
    result_dir = os.path.join('feature_maps', name, config['model_name'].lower())
    os.makedirs(result_dir, exist_ok=True)
    feature_map = feature_maps[name].squeeze().cpu()
    for i in range(feature_map.size()[0]):
        plt.imshow(feature_map[i])
        plt.show()

    # Classifier feature map 저장
    result_dir = os.path.join('feature_maps', 'classifier', config['model_name'].lower())
    os.makedirs(result_dir, exist_ok=True)
    for i in range(mask_pred.size()[0]):
        plt.imsave(os.path.join(result_dir, '{}.png'.format(i + 1)), mask_pred[i])
