import os

import torch.nn as nn
import torch.utils.data
import torch.utils.tensorboard
import tqdm

import eval
import utils

if __name__ == '__main__':
    # 0. Load config
    config = utils.load_config()
    print('Activated model: {}'.format(config['model']))

    # 1. Dataset
    dataset = utils.Cityscapes(config)
    _, trainloader, _, testloader = dataset.set_cityscapes()

    # 2. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = utils.get_model(config).to(device)

    # 3. Loss function, optimizer, lr scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = utils.get_optimizer(config, model)
    scheduler = utils.get_scheduler(config, optimizer)

    # 4. Tensorboard
    writer = torch.utils.tensorboard.SummaryWriter(os.path.join('runs', config['model']))
    writer.add_graph(model, trainloader.__iter__().__next__()[0].to(device))

    # 5. Train and evaluate
    log_loss = tqdm.tqdm(total=0, position=2, bar_format='{desc}', leave=False)
    prev_miou = 0.0
    prev_val_loss = 0.0
    for epoch in tqdm.tqdm(range(config[config['model']]['epoch']), desc='Epoch'):
        model.train()

        for batch_idx, (image, target) in enumerate(tqdm.tqdm(trainloader, desc='Train', leave=False)):
            # mask에 255를 곱하여 0~1 사이의 값을 0~255 값으로 변경 + 채널 차원 제거
            target.mul_(255).squeeze_(dim=1)

            image, target = image.to(device), target.to(device, dtype=torch.int64)

            # 순전파 + 역전파 + 최적화
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # 손실값 출력
            log_loss.set_description_str('Loss: {:.4f}'.format(loss.item()))

            # Tensorboard에 학습 과정 기록
            writer.add_scalar('Train Loss', loss.item(), len(trainloader) * epoch + batch_idx)

        # 모델 평가
        val_loss, _, miou, _ = eval.evaluate(model, testloader, criterion,
                                             config[config['model']]['num_classes'], device)
        writer.add_scalar('Validation Loss', val_loss, epoch)
        writer.add_scalar('mIoU', miou, epoch)

        # lr scheduler의 step을 진행
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        scheduler.step(val_loss)

        # 가장 마지막 epoch의 모델을 저장
        os.makedirs('weights', exist_ok=True)
        torch.save(model.state_dict(), os.path.join('weights', '{}_last.pth'.format(config['model'])))

        # Best mIoU를 가진 모델을 저장
        if miou > prev_miou:
            torch.save(model.state_dict(), os.path.join('weights', '{}_best.pth'.format(config['model'])))
            prev_miou = miou

        # Best val_loss를 가진 모델을 저장
        if val_loss > prev_val_loss:
            torch.save(model.state_dict(), os.path.join('weights', '{}_val_best.pth'.format(config['model'])))
            prev_val_loss = val_loss
    writer.close()
