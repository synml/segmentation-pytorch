import os

import torch.nn as nn
import torch.utils.data
import torch.utils.tensorboard
import tqdm

import model.unet
import model.proposed
import utils.utils
import evaluate


if __name__ == '__main__':
    # 설정 불러오기
    ini_file = 'model/unet.ini'
    config, section = utils.utils.load_config(ini_file)
    print('{}를 불러왔습니다.'.format(ini_file.split('/')[-1]))

    # 장치, 시각 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터셋 설정
    trainset, trainloader, testset, testloader = utils.utils.init_cityscapes_dataset(config)

    # 모델 설정
    if section == 'unet':
        model = model.unet.UNet(3, config['num_classes']).to(device)
    elif section == 'proposed':
        model = model.proposed.Proposed(3, config['num_classes']).to(device)

    # Loss Function, Optimizer 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # learning rate scheduler 설정
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=0.0001)

    # Tensorboard 설정
    writer = torch.utils.tensorboard.SummaryWriter(os.path.join('runs', section))

    # 모델 그래프를 Tensorboard에 저장
    writer.add_graph(model, trainloader.__iter__().__next__()[0].to(device))

    # 현재 배치 손실값을 출력하는 tqdm 설정
    log_loss = tqdm.tqdm(total=0, position=2, bar_format='{desc}', leave=False)

    # Train에 필요한 변수들을 설정
    prev_miou = 0.0
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    model_name = model.__module__.split('.')[-1]

    # Train
    for epoch in tqdm.tqdm(range(config['epoch']), desc='Epoch'):
        model.train()

        # 1 epoch의 각 배치에서 처리하는 코드
        for batch_idx, (images, masks) in enumerate(tqdm.tqdm(trainloader, desc='Batch', leave=False)):
            step = len(trainloader) * epoch + batch_idx

            # mask에 255를 곱하여 0~1 사이의 값을 0~255 값으로 변경 + 채널 차원 제거
            masks = torch.mul(masks, 255)
            masks = torch.squeeze(masks, dim=1)

            # 이미지와 정답 정보를 GPU로 복사
            images = images.to(device)
            masks = masks.to(device, dtype=torch.long)

            # 변화도(Gradient) 매개변수를 0으로 만들기
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화
            masks_pred = model(images)
            loss = criterion(masks_pred, masks)
            loss.backward()
            optimizer.step()

            # 손실값 출력
            log_loss.set_description_str('Loss: {:.4f}'.format(loss.item()))

            # Tensorboard에 학습 과정 기록
            writer.add_scalar('Train loss', loss.item(), step)

        # 모델을 평가
        miou, _, val_loss, _ = evaluate.evaluate(model, testloader, device, config['num_classes'])

        # Tensorboard에 값 기록
        writer.add_scalar('mIoU', miou, epoch)
        writer.add_scalar('Val loss', val_loss, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # lr scheduler의 step을 진행
        scheduler.step(val_loss)

        # checkpoint file 저장
        torch.save(model.state_dict(), os.path.join(save_dir, '{}_{}.pth'.format(model_name, epoch)))

        # Best mIoU를 가진 모델을 저장
        if miou > prev_miou:
            torch.save(model.state_dict(), os.path.join(save_dir, '{}_best.pth'.format(model_name)))
            prev_miou = miou

    writer.close()
