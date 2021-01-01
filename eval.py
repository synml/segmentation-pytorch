import csv
import os
import time
from typing import List

import numpy as np
import sklearn.metrics
import torch.nn.functional as F
import torch.utils.data
import tqdm

import utils.utils
import utils.datasets


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


def evaluate(model, testloader, num_classes: int, device):
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
        images, masks = images.to(device), masks.to(device, dtype=torch.long)

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

    return val_loss, iou, miou, fps


if __name__ == '__main__':
    # 0. Load config
    model_name, config = utils.utils.load_config()
    print('Activated model: {}'.format(model_name))

    # 1. Dataset
    _, _, testset, testloader = utils.utils.init_cityscapes_dataset(config)

    # 2. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = utils.utils.get_model(model_name, 3, config['num_classes'], config['pretrained_weights']).to(device)

    # 모델 평가
    val_loss, iou, miou, fps = evaluate(model, testloader, config['num_classes'], device)

    # 평가 결과를 csv 파일로 저장
    os.makedirs('result', exist_ok=True)
    with open(os.path.join('result', '{}.csv'.format(model_name)), mode='w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')

        writer.writerow(['Class Number', 'Class Name', 'IoU'])
        for class_num, iou_value in enumerate(iou, start=1):
            writer.writerow([class_num, utils.datasets.Cityscapes.class_names_short[class_num], iou_value])
        writer.writerow(['mIoU', miou, ' '])
        writer.writerow(['Validation loss', val_loss, ' '])
        writer.writerow(['FPS', fps, ' '])
    print('Saved evaluation result')
