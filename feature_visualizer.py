import os

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data
import tqdm

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

    # 이미지 불러오기
    image_number = input('Enter the image number of the dataset>>> ')
    if image_number == '':
        image_number = 0
    else:
        image_number = int(image_number)
    image, _ = testset[image_number]
    image = image.unsqueeze(0).to(device)

    # 모델의 각 계층에 특징맵을 받아오는 hook을 등록
    feature_maps = {}
    if config['model_name'] == 'UNet':
        model.encode1.register_forward_hook(get_feature_maps(feature_maps, 'encode1'))
        model.encode2.register_forward_hook(get_feature_maps(feature_maps, 'encode2'))
        model.encode3.register_forward_hook(get_feature_maps(feature_maps, 'encode3'))
        model.encode4.register_forward_hook(get_feature_maps(feature_maps, 'encode4'))
        model.encode_end.register_forward_hook(get_feature_maps(feature_maps, 'encode_end'))
        model.decode4.register_forward_hook(get_feature_maps(feature_maps, 'decode4'))
        model.decode3.register_forward_hook(get_feature_maps(feature_maps, 'decode3'))
        model.decode2.register_forward_hook(get_feature_maps(feature_maps, 'decode2'))
        model.decode1.register_forward_hook(get_feature_maps(feature_maps, 'decode1'))
        model.classifier.register_forward_hook(get_feature_maps(feature_maps, 'classifier'))
    elif config['model_name'] == 'Proposed':
        model.initial_conv.register_forward_hook(get_feature_maps(feature_maps, 'initial_conv'))
        model.encode1.register_forward_hook(get_feature_maps(feature_maps, 'encode1'))
        model.encode2.register_forward_hook(get_feature_maps(feature_maps, 'encode2'))
        model.encode3.register_forward_hook(get_feature_maps(feature_maps, 'encode3'))
        model.encode4.register_forward_hook(get_feature_maps(feature_maps, 'encode4'))
        model.aspp.register_forward_hook(get_feature_maps(feature_maps, 'aspp'))
        model.decode3.register_forward_hook(get_feature_maps(feature_maps, 'decode3'))
        model.decode2.register_forward_hook(get_feature_maps(feature_maps, 'decode2'))
        model.decode1.register_forward_hook(get_feature_maps(feature_maps, 'decode1'))
        model.classifier.register_forward_hook(get_feature_maps(feature_maps, 'classifier'))
    elif config['model_name'] == 'Backbone':
        model.initial_conv.register_forward_hook(get_feature_maps(feature_maps, 'initial_conv'))
        model.layer1.register_forward_hook(get_feature_maps(feature_maps, 'layer1'))
        model.layer2.register_forward_hook(get_feature_maps(feature_maps, 'layer2'))
        model.layer3.register_forward_hook(get_feature_maps(feature_maps, 'layer3'))
        model.layer4.register_forward_hook(get_feature_maps(feature_maps, 'layer4'))

    # 예측
    with torch.no_grad():
        mask_pred = model(image)

    # 각 계층의 feature maps 저장
    for layer in tqdm.tqdm(feature_maps.keys(), desc='Saving'):
        result_dir = os.path.join('feature_maps', config['model_name'], layer)
        os.makedirs(result_dir, exist_ok=True)
        feature_map = feature_maps[layer].squeeze().cpu()
        if layer == 'classifier':
            feature_map = F.log_softmax(feature_map, dim=0)

        for i in tqdm.tqdm(range(feature_map.size()[0]), desc='Channels', leave=False):
            plt.imsave(os.path.join(result_dir, '{}.png'.format(i + 1)), feature_map[i])
