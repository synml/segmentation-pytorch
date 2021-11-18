import os

import cv2
import numpy as np
from PIL import Image
import pytorch_grad_cam
import torch
import torchvision
import tqdm

import utils


def draw_cam_on_image(image: torch.Tensor, mask: np.ndarray, colormap=cv2.COLORMAP_JET) -> torch.Tensor:
    assert torch.min(image) >= 0 and torch.max(image) <= 1, 'Input image should in the range [0, 1]'

    heatmap = cv2.applyColorMap(np.uint8(mask * 255), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1) / 255

    cam = image + heatmap
    cam /= torch.max(cam)
    return cam


def cam_get_loss(output, target_category):
    loss = 0
    for i in range(len(target_category)):
        loss = loss + output[i, target_category, :, :].mean()
    return loss


if __name__ == '__main__':
    # Load cfg and create components builder
    cfg = utils.builder.load_cfg()
    builder = utils.builder.Builder(cfg)

    # 1. Dataset
    valset, _ = builder.build_dataset('val')

    # 2. Model
    model = builder.build_model(valset.num_classes, pretrained=True)
    model.eval()
    model_name = cfg['model']['name']
    amp_enabled = cfg['model']['amp_enabled']
    print(f'Activated model: {model_name}')

    # 이미지 불러오기
    image_number = input('Enter the image number of the dataset>>> ')
    if image_number == '':
        image_number = 253
    image_number = int(image_number) - 1
    image, _ = valset[image_number]
    image.unsqueeze_(0)

    # Class activation map을 생성할 계층을 지정
    use_cuda = torch.cuda.is_available()
    cam_layers = {
        'dappm': pytorch_grad_cam.GradCAMPlusPlus(model, target_layer=model.dappm, use_cuda=use_cuda),
        'decoder': pytorch_grad_cam.GradCAMPlusPlus(model, target_layer=model.decoder, use_cuda=use_cuda),
    }

    # Class activation map 생성
    for layer, cam in tqdm.tqdm(cam_layers.items(), desc='Saving CAM'):
        result_dir = os.path.join('cam', model_name, layer)
        os.makedirs(result_dir, exist_ok=True)

        for target_category in tqdm.tqdm(range(valset.num_classes), desc='Classes', leave=False):
            cam_mask: np.ndarray = cam(image, target_category)[0, :]
            cam_on_image = draw_cam_on_image(
                torchvision.transforms.ToTensor()(Image.open(valset.images[image_number]).convert('RGB')),
                cam_mask
            )
            torchvision.utils.save_image(
                cam_on_image,
                os.path.join(result_dir, f'{target_category}_{valset.class_names[target_category]}.jpg')
            )
