import os

import pytorch_grad_cam.utils.image
import torch.cuda
import torchvision.utils
import tqdm

import utils


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
        image_number = 0
    else:
        image_number = int(image_number)
    image, _ = valset[image_number]
    image.unsqueeze_(0)

    # Class activation map을 생성할 계층을 지정
    gradcam_layers = {
        'backbone': pytorch_grad_cam.GradCAM(model, target_layer=model.backbone, use_cuda=torch.cuda.is_available()),
        'aspp': pytorch_grad_cam.GradCAM(model, target_layer=model.aspp, use_cuda=torch.cuda.is_available()),
        'decoder': pytorch_grad_cam.GradCAM(model, target_layer=model.decoder, use_cuda=torch.cuda.is_available()),
    }

    # Class activation map 생성
    for layer, gradcam in tqdm.tqdm(gradcam_layers.items(), desc='Saving CAM'):
        result_dir = os.path.join('cam', model_name, layer)
        os.makedirs(result_dir, exist_ok=True)
        for target_category in range(valset.num_classes):
            cam = gradcam(image, target_category)
            visualization = pytorch_grad_cam.utils.image.show_cam_on_image(
                image.squeeze_(0).permute(1, 2, 0).numpy(), cam[0, :]
            )
            torchvision.utils.save_image(torch.from_numpy(visualization),
                                         os.path.join(result_dir, f'{target_category}.png'))
