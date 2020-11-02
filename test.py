import configparser
import csv
import os
import time

import numpy as np
import torch.nn.functional as F
import torch.utils.data
import torchvision
import tqdm

import model.unet


# IoU (Intersection over Union)를 계산한다.
def calc_iou(gt_batch: torch.Tensor, pred_batch: torch.Tensor, batch_size: int, num_classes: int):
    iou = np.zeros(num_classes)

    # 각 배치 처리
    for idx in range(batch_size):
        # 채널 차원 제거, ndarrary로 변환
        gt = gt_batch[idx].squeeze().cpu().numpy()
        pred = pred_batch[idx].squeeze().cpu().numpy()

        # 2차원 행렬을 1차원 벡터로 변환
        gt_count = gt.reshape(-1)
        pred_count = pred.reshape(-1)

        # 카테고리 행렬 생성
        category = gt_count * num_classes + pred_count

        # 혼동 행렬 생성
        confusion_matrix = np.bincount(category, minlength=num_classes**2).reshape((num_classes, num_classes))

        # 클래스 별 IoU 계산 (intersection / union = TP / (TP + FP + FN))
        for i in range(num_classes):
            intersection = 0
            union = 0

            # intersection, union 계산
            for k in range(num_classes):
                union += confusion_matrix[i][k]  # 횡으로 덧셈
                # 같은 원소를 가리킬 때, intersection을 구함
                if i == k:
                    intersection = confusion_matrix[i][k]
                    continue
                union += confusion_matrix[k][i]  # 종으로 덧셈

            # 클래스 별 IoU = intersection / union
            if union != 0:
                iou[i] += intersection / union

    return iou


def evaluate(model, testloader, device, num_classes: int):
    model.eval()

    # Evaluate
    iou = np.zeros(num_classes)
    total_loss = 0
    entire_time = 0
    for images, masks in tqdm.tqdm(testloader, desc='Eval', leave=False):
        # mask에 255를 곱하여 0~1 사이의 값을 0~255 값으로 변경 + 채널 차원 제거
        masks = torch.mul(masks, 255)
        masks = torch.squeeze(masks, dim=1)

        # 이미지와 정답 정보를 GPU로 복사
        images = images.to(device)
        masks = masks.to(device, dtype=torch.long)

        # 예측
        start_time = time.time()
        with torch.no_grad():
            masks_pred = model(images)
        entire_time += time.time() - start_time

        # validation loss를 모두 합침
        total_loss += F.cross_entropy(masks_pred, masks, reduction='sum').item()

        # Segmentation map 만들기
        masks_pred = F.softmax(masks_pred, dim=1)
        masks_pred = torch.argmax(masks_pred, dim=1, keepdim=True)

        # 배치당 IoU를 계산
        iou_batch = calc_iou(masks, masks_pred, testloader.batch_size, num_classes)
        for i in range(num_classes):
            iou[i] += iou_batch[i]

    # mIoU를 계산
    miou = np.mean(iou)

    # 평균 validation loss 계산
    val_loss = total_loss / (len(testloader) * testloader.batch_size)

    # 추론 시간과 fps를 계산
    inference_time = entire_time / (len(testloader) * testloader.batch_size)
    fps = 1 / inference_time

    # 추론 시간을 miliseconds 단위로 설정
    inference_time *= 1000

    return miou, val_loss, inference_time, fps


if __name__ == '__main__':
    parser = configparser.ConfigParser()
    parser.read('model/unet.ini', encoding='utf-8')
    config = {
        'batch_size': parser.getint('UNet', 'batch_size'),
        'image_size': parser.getint('UNet', 'image_size'),
        'num_classes': parser.getint('UNet', 'num_classes'),
        'num_workers': parser.getint('UNet', 'num_workers'),
        'pretrained_weights': parser['UNet']['pretrained_weights'],
        'result_dir': 'csv/'
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터셋 설정
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(config['image_size']),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    target_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(config['image_size']),
        torchvision.transforms.ToTensor(),
    ])
    testset = torchvision.datasets.Cityscapes(root='../../data/cityscapes',
                                              split='val',
                                              mode='fine',
                                              target_type='semantic',
                                              transform=transform,
                                              target_transform=target_transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=config['batch_size'],
                                             shuffle=False,
                                             num_workers=config['num_workers'],
                                             pin_memory=True)

    # 모델 설정
    model = model.unet.UNet(3, config['num_classes']).to(device)
    model.load_state_dict(torch.load(config['pretrained_weights']))

    # 모델 평가
    miou, val_loss, inference_time, fps = evaluate(model, testloader, device, config['num_classes'])

    # mIoU, Validation loss, Inference time, FPS 출력
    print('mIoU: {:.4f}'.format(miou))
    print('Validation loss: {:.4f}'.format(val_loss))
    print('Inference time (ms): {:.02f}'.format(inference_time))
    print('FPS: {:.02f}'.format(fps))

    # Validation loss, Inference time, FPS를 csv 파일로 저장
    os.makedirs(config['result_dir'], exist_ok=True)
    now = time.strftime('%y%m%d_%H%M%S', time.localtime(time.time()))
    filename = '{}_{}.csv'.format(model.__module__.lower(), now)
    with open(os.path.join(config['result_dir'], filename), mode='w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerow(['Validation loss', val_loss])
        writer.writerow(['Inference time (ms)', inference_time])
        writer.writerow(['FPS', fps])
    print('평가 결과를 csv 파일로 저장했습니다.')
