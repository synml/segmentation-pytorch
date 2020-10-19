import argparse
import csv
import os
import time

import torch
import torch.utils.data
import numpy as np
import tqdm

import model.unet


def eval(model, testloader, device):
    model.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--image_size', default=(1024, 512))
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument("--weights", type=str)
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 설정
    model = model.unet.UNet(3, 19).to(device)
    model.load_state_dict(torch.load(args.weights))
