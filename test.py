import csv
import os
import time

import numpy as np
import torch.nn.functional as F
import torch.utils.data
import tqdm

import model.unet
import model.proposed
import utils.dataset
import utils.utils


# IoU (Intersection over Union)를 계산한다.
# reduction='mean': IoU는 batch에 대한 산술평균 값
# reduction='sum': IoU는 batch의 각 이미지에서 계산된 IoU 값의 합
def calc_iou(gt_batch: torch.Tensor, pred_batch: torch.Tensor, num_classes: int, reduction='mean'):
    assert reduction == 'mean' or reduction == 'sum'

    # 1batch의 이미지에 대한 iou를 합하여 저장
    iou = np.zeros(num_classes)

    # 1batch에 포함된 각 이미지의 iou를 계산
    for idx in range(gt_batch.shape[0]):
        # 채널 차원 제거, ndarrary로 변환
        gt = gt_batch[idx].squeeze().cpu().numpy()
        pred = pred_batch[idx].squeeze().cpu().numpy()

        # 2차원 행렬을 1차원 벡터로 변환
        gt_count = gt.reshape(-1)
        pred_count = pred.reshape(-1)

        # 카테고리 행렬 생성
        category = gt_count * num_classes + pred_count

        # 혼동 행렬 생성
        confusion_matrix = np.bincount(category, minlength=num_classes ** 2).reshape((num_classes, num_classes))

        # 각 이미지의 IoU를 계산 (intersection / union = TP / (TP + FP + FN))
        for i in range(num_classes):
            intersection = 0
            union = 0

            # intersection과 union을 계산
            for k in range(num_classes):
                union += confusion_matrix[i][k]  # 횡으로 덧셈
                # 같은 원소를 가리킬 때, intersection을 구함
                if i == k:
                    intersection = confusion_matrix[i][k]
                    continue
                union += confusion_matrix[k][i]  # 종으로 덧셈

            # IoU = intersection / union (ZeroDivisionError 방지)
            if union != 0:
                iou[i] += intersection / union

    # reduction이 mean이면 batch에 대한 산술평균을 수행
    if reduction == 'mean':
        return iou / gt_batch.shape[0]

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
        total_loss += F.cross_entropy(masks_pred, masks).item()

        # Segmentation map 만들기
        masks_pred = F.softmax(masks_pred, dim=1)
        masks_pred = torch.argmax(masks_pred, dim=1, keepdim=True)

        # 각 batch의 IoU를 계산
        iou_batch = calc_iou(masks, masks_pred, num_classes, reduction='sum')
        for i in range(num_classes):
            iou[i] += iou_batch[i]

    # 데이터셋 전체의 IoU를 계산 (백분율 단위)
    iou = iou / len(testloader.dataset) * 100
    iou = iou[:-1]

    # mIoU를 계산 (백분율 단위)
    miou = np.mean(iou)

    # 평균 validation loss 계산
    val_loss = total_loss / len(testloader)

    # 추론 시간과 fps를 계산
    inference_time = entire_time / len(testloader.dataset)
    fps = 1 / inference_time

    # 추론 시간을 miliseconds 단위로 설정
    inference_time *= 1000

    return miou, iou, val_loss, fps


if __name__ == '__main__':
    # 설정 불러오기
    ini_file = 'model/unet.ini'
    config, section = utils.utils.load_config(ini_file)

    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터셋 설정
    _, _, testset, testloader = utils.utils.init_dataset(config)

    # 모델 설정
    if section == 'unet':
        model = model.unet.UNet(3, config['num_classes']).to(device)
    elif section == 'proposed':
        model = model.proposed.Proposed(3, config['num_classes']).to(device)
    model.load_state_dict(torch.load(config['pretrained_weights']))

    # 모델 평가
    miou, iou, val_loss, fps = evaluate(model, testloader, device, config['num_classes'])

    # mIoU, Validation loss, Inference time, FPS 출력
    print('mIoU: {:.4f}'.format(miou))
    print('Validation loss: {:.4f}'.format(val_loss))
    print('FPS: {:.02f}'.format(fps))

    # Validation loss, Inference time, FPS를 csv 파일로 저장
    os.makedirs('csv', exist_ok=True)
    now = time.strftime('%y%m%d_%H%M%S', time.localtime(time.time()))
    filename = '{}_{}.csv'.format(model.__module__.split('.')[-1], now)
    with open(os.path.join(config['result_dir'], filename), mode='w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')

        writer.writerow(['Class Number', 'Class Name', 'IoU'])
        for class_num, iou_value in enumerate(iou):
            writer.writerow([class_num, utils.dataset.class_names[class_num], iou_value])
        writer.writerow(['mIoU', miou, ' '])
        writer.writerow(['Validation loss', val_loss, ' '])
        writer.writerow(['FPS', fps, ' '])
    print('평가 결과를 csv 파일로 저장했습니다.')
