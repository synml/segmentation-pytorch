import os

import torch.utils.data
import torchvision
import tqdm

import utils

if __name__ == '__main__':
    # Load cfg and create components builder
    cfg = utils.builder.load_cfg()
    builder = utils.builder.Builder(cfg)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Dataset
    testset, testloader = builder.build_dataset('test')

    # 2. Model
    model = builder.build_model(testset.num_classes, pretrained=True).to(device)
    model.eval()
    model_name = cfg['model']['name']
    amp_enabled = cfg['model']['amp_enabled']
    print(f'Activated model: {model_name}')

    # Load cities and image paths
    cities = sorted(os.listdir(testset.images_dir))
    image_paths = []
    for image_path in testset.images:
        image_path = image_path.replace('\\', '/').split('/')[-2:]
        image_path = os.path.join(*image_path)
        image_paths.append(image_path)

    # Make result directories
    step = 0
    result_dir = os.path.join('submission', model_name.lower())
    os.makedirs(result_dir, exist_ok=True)
    for city in cities:
        os.makedirs(os.path.join(result_dir, city), exist_ok=True)

    # Save segmentation results
    for images, targets in tqdm.tqdm(testloader, desc='Create submission'):
        images, targets = images.to(device), targets.to(device)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            with torch.no_grad():
                outputs = model(images)
                outputs = torch.argmax(outputs, dim=1)

        outputs = testset.decode_segmap_to_test_id(outputs)

        # process per 1 batch
        for i in range(targets.shape[0]):
            torchvision.utils.save_image(outputs[i], os.path.join(result_dir, image_paths[step]))
            step += 1
