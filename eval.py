import csv
import os
import time
from typing import List

import numpy as np
import sklearn.metrics
import torch.nn.functional as F
import torch.utils.data
import tqdm

import model.unet
import model.proposed
import utils.datasets
import utils.utils


class EvaluationMetrics:
    def __init__(self, labels: List[int]):
        self.labels = labels
        self.confusion_matrix = np.zeros((len(labels), len(labels)))

    def update_matrix(self, gt_batch: torch.Tensor, pred_batch: torch.Tensor):
        assert gt_batch.shape[0] == pred_batch.shape[0]

        gt = torch.flatten(gt_batch, start_dim=1).cpu().numpy()
        pred = torch.flatten(pred_batch, start_dim=1).cpu().numpy()

        # 1 배치단위 처리
        for i in range(gt_batch.shape[0]):
            self.confusion_matrix += sklearn.metrics.confusion_matrix(gt[i], pred[i], labels=self.labels)

    def get_scores(self, ignore_first_label=False, ignore_last_label=False):
        if ignore_first_label:
            self.confusion_matrix = self.confusion_matrix[1:, 1:]
        if ignore_last_label:
            last_label = self.confusion_matrix.shape[0] - 1
            self.confusion_matrix = self.confusion_matrix[:last_label, :last_label]

        iou = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=0) +
                                                self.confusion_matrix.sum(axis=1) -
                                                np.diag(self.confusion_matrix)) * 100
        miou = np.mean(iou)
        return iou, miou


def evaluate(model, testloader, device, num_classes: int):
    model.eval()

    # Evaluate
    metrics = EvaluationMetrics(list(range(num_classes)))
    val_loss = 0
    inference_time = 0
    for images, masks in tqdm.tqdm(testloader, desc='Eval', leave=False):
        # mask에 255를 곱하여 0~1 사이의 값을 0~255 값으로 변경 + 채널 차원 제거
        masks = torch.mul(masks, 255)
        masks = torch.squeeze(masks, dim=1)

        # 이미지와 정답 정보를 GPU로 복사
        images = images.to(device)
        masks = masks.to(device, dtype=torch.long)

        # 예측
        with torch.no_grad():
            start_time = time.time()
            masks_pred = model(images)
            inference_time += time.time() - start_time

        # validation loss를 모두 합침
        val_loss += F.cross_entropy(masks_pred, masks, reduction='sum').item()

        # Segmentation map 만들기
        masks_pred = F.log_softmax(masks_pred, dim=1)
        masks_pred = torch.argmax(masks_pred, dim=1, keepdim=True)

        # 혼동행렬 업데이트
        metrics.update_matrix(masks, masks_pred)

    # 평가 점수 가져오기
    iou, miou = metrics.get_scores(ignore_first_label=True)

    # 평균 validation loss 계산
    val_loss /= len(testloader.dataset)

    # 추론 시간과 fps를 계산 (추론 시간: ms)
    inference_time /= len(testloader.dataset)
    fps = 1 / inference_time

    return iou, miou, val_loss, fps


if __name__ == '__main__':
    # 설정 불러오기
    ini_file = 'model/unet.ini'
    config, section = utils.utils.load_config(ini_file)
    print('{}를 불러왔습니다.'.format(ini_file.split('/')[-1]))

    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터셋 설정
    _, _, testset, testloader = utils.utils.init_cityscapes_dataset(config)

    # 모델 설정
    if section == 'unet':
        model = model.unet.UNet(3, config['num_classes']).to(device)
    elif section == 'proposed':
        model = model.proposed.Proposed(3, config['num_classes']).to(device)
    model.load_state_dict(torch.load(config['pretrained_weights']))

    # 모델 평가
    iou, miou, val_loss, fps = evaluate(model, testloader, device, config['num_classes'])

    # 평가 결과를 csv 파일로 저장
    result_dir = 'result'
    os.makedirs(result_dir, exist_ok=True)
    filename = '{}.csv'.format(model.__module__.split('.')[-1])
    with open(os.path.join(result_dir, filename), mode='w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')

        writer.writerow(['Class Number', 'Class Name', 'IoU'])
        for class_num, iou_value in enumerate(iou, start=1):
            writer.writerow([class_num, utils.datasets.Cityscapes.class_names_short[class_num], iou_value])
        writer.writerow(['mIoU', miou, ' '])
        writer.writerow(['Validation loss', val_loss, ' '])
        writer.writerow(['FPS', fps, ' '])
    print('평가 결과를 csv 파일로 저장했습니다.')
