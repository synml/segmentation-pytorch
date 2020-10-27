import configparser
import csv
import os
import time

import torch.nn.functional as F
import torch.utils.data
import torchvision
import tqdm

import model.unet


def evaluate(model, testloader, device):
    model.eval()

    # Evaluate
    total_loss = 0
    entire_time = 0
    for images, masks in tqdm.tqdm(testloader, desc='Batch', leave=False):
        # 이미지와 정답 정보를 GPU로 복사
        images = images.to(device)
        masks = masks.to(device)

        # 순전파
        start_time = time.time()
        with torch.no_grad():
            masks_pred = model(images)
        entire_time += time.time() - start_time

        # validation loss를 모두 합침
        total_loss += F.mse_loss(masks_pred, masks, reduction='sum').item()

    # 평균 validation loss 계산
    val_loss = total_loss / testloader.__len__()

    # 추론 시간과 fps를 계산
    inference_time = entire_time / testloader.__len__()
    fps = 1 / inference_time

    # 추론 시간을 miliseconds 단위로 설정
    inference_time *= 1000

    return val_loss, inference_time, fps


if __name__ == '__main__':
    parser = configparser.ConfigParser()
    parser.read('u-net.ini', encoding='utf-8')
    config = {
        'batch_size': parser.getint('U-Net', 'batch_size'),
        'epoch': parser.getint('U-Net', 'epoch'),
        'image_size': parser.getint('U-Net', 'image_size'),
        'lr': parser.getfloat('U-Net', 'lr'),
        'num_workers': parser.getint('U-Net', 'num_workers'),
        'pretrained_weights': parser.getint('U-Net', 'pretrained_weights')
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터셋 설정
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(config['image_size']),
        torchvision.transforms.ToTensor()
    ])
    testset = torchvision.datasets.Cityscapes(root='../../data/cityscapes',
                                              split='val',
                                              mode='fine',
                                              target_type='semantic',
                                              transform=transform,
                                              target_transform=torchvision.transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=config['batch_size'],
                                             shuffle=True,
                                             num_workers=config['num_workers'],
                                             pin_memory=True)

    # 모델 설정
    model = model.unet.UNet(3, 19).to(device)
    model.load_state_dict(torch.load(config['pretrained_weights']))

    # 모델 평가
    val_loss, inference_time, fps = evaluate(model, testloader, device)

    # Validation loss, Inference time, FPS 출력
    print('Validation loss: {:.4f}'.format(val_loss))
    print('Inference time (ms): {:.02f}'.format(inference_time))
    print('FPS: {:.02f}'.format(fps))

    # Validation loss, Inference time, FPS를 csv 파일로 저장
    os.makedirs('csv', exist_ok=True)
    now = time.strftime('%y%m%d_%H%M%S', time.localtime(time.time()))
    with open('csv/test{}.csv'.format(now), mode='w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerow(['Validation loss', val_loss])
        writer.writerow(['Inference time (ms)', inference_time])
        writer.writerow(['FPS', fps])
    print('평가 결과를 csv 파일로 저장했습니다.')
