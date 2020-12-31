import os

import torch.nn as nn
import torch.utils.data
import torch.utils.tensorboard
import tqdm

import model.unet
import model.proposed
import utils.utils
import eval


if __name__ == '__main__':
    # 설정 불러오기
    ini_file = 'model/unet.ini'
    config, section = utils.utils.load_config(ini_file)
    print('{}를 불러왔습니다.'.format(ini_file.split('/')[-1]))

    # 1. Dataset
    trainset, trainloader, testset, testloader = utils.utils.init_cityscapes_dataset(config)

    # 2. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if section == 'unet':
        model = model.unet.UNet(3, config['num_classes']).to(device)
    elif section == 'proposed':
        model = model.proposed.Proposed(3, config['num_classes']).to(device)

    # 3. Loss function, optimizer, lr scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=0.0001)

    # 4. Tensorboard
    writer = torch.utils.tensorboard.SummaryWriter(os.path.join('runs', section))
    writer.add_graph(model, trainloader.__iter__().__next__()[0].to(device))

    # 5. Train and test
    log_loss = tqdm.tqdm(total=0, position=2, bar_format='{desc}', leave=False)
    prev_miou = 0.0
    for epoch in tqdm.tqdm(range(config['epoch']), desc='Epoch'):
        model.train()

        for batch_idx, (images, masks) in enumerate(tqdm.tqdm(trainloader, desc='Train', leave=False)):
            # mask에 255를 곱하여 0~1 사이의 값을 0~255 값으로 변경 + 채널 차원 제거
            masks = torch.mul(masks, 255)
            masks = torch.squeeze(masks, dim=1)

            # 이미지와 정답 정보를 GPU로 복사
            images, masks = images.to(device), masks.to(device, dtype=torch.long)

            # 순전파 + 역전파 + 최적화
            optimizer.zero_grad()
            masks_pred = model(images)
            loss = criterion(masks_pred, masks)
            loss.backward()
            optimizer.step()

            # 손실값 출력
            log_loss.set_description_str('Loss: {:.4f}'.format(loss.item()))

            # Tensorboard에 학습 과정 기록
            writer.add_scalar('Train Loss', loss.item(), len(trainloader) * epoch + batch_idx)

        # 모델을 평가
        val_loss, _, miou, _ = eval.evaluate(model, testloader, device, config['num_classes'])

        # Tensorboard에 값 기록
        writer.add_scalar('Validation Loss', val_loss, epoch)
        writer.add_scalar('mIoU', miou, epoch)
        writer.add_text('lr', 'lr changed: {}'.format(optimizer.param_groups[0]['lr']), epoch)

        # lr scheduler의 step을 진행
        scheduler.step(val_loss)

        # Best mIoU를 가진 모델을 저장
        if miou > prev_miou:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join('checkpoints', '{}_best.pth'.format(model.__module__.split('.')[-1])))
            prev_miou = miou
    writer.close()
