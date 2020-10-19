import configparser
import os
import time

import torch.nn as nn
import torch.utils.data
import torch.utils.tensorboard
import torchvision
import tqdm

import model.unet
import test

if __name__ == '__main__':
    parser = configparser.ConfigParser()
    parser.read('config.ini', encoding='utf-8')
    config = {
        'batch_size': parser.getint('U-Net', 'batch_size'),
        'epoch': parser.getint('U-Net', 'epoch'),
        'image_size': parser.getint('U-Net', 'image_size'),
        'lr': parser.getfloat('U-Net', 'lr'),
        'num_workers': parser.getint('U-Net', 'num_workers')
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋 설정
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(config['image_size']),
        torchvision.transforms.ToTensor()
    ])
    trainset = torchvision.datasets.Cityscapes(root='../../data/cityscapes',
                                               split='train',
                                               mode='fine',
                                               target_type='semantic',
                                               transform=transform,
                                               target_transform=torchvision.transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=config['batch_size'],
                                              shuffle=True,
                                              num_workers=config['num_workers'],
                                              pin_memory=True)
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
    model = model.unet.UNet(3, 19)
    model.to(device)

    # Loss Function, Optimizer 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # learning rate scheduler 설정
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    # Tensorboard 설정
    writer = torch.utils.tensorboard.SummaryWriter()

    # 현재 배치 손실값을 출력하는 tqdm 설정
    log_loss = tqdm.tqdm(total=0, position=2, bar_format='{desc}', leave=False)

    # Train
    for epoch in tqdm.tqdm(range(config['epoch']), desc='Epoch'):
        model.train()

        # 1 epoch의 각 배치에서 처리하는 코드
        for batch_idx, (images, masks) in enumerate(tqdm.tqdm(trainloader, desc='Batch', leave=False)):
            step = len(trainloader) * epoch + batch_idx

            # 이미지와 정답 정보를 GPU로 복사
            images = images.to(device)
            masks = masks.to(device)

            # 변화도(Gradient) 매개변수를 0으로 만들기
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화
            masks_pred = model(images)
            loss = criterion(masks_pred, masks)
            loss.backward()
            optimizer.step()

            # 손실값 출력
            log_loss.set_description_str('Loss: {:.4f}'.format(loss.item()))

            # Tensorboard에 loss 기록
            writer.add_scalar('Train loss', loss.item(), step)

        # 모델을 평가
        val_loss, _, _ = test.evaluate(model, testloader, device)

        # Tensorboard에 평가 결과 기록
        writer.add_scalar('Val loss', val_loss, epoch)

        # lr scheduler의 step을 진행
        scheduler.step(val_loss)

        # checkpoint file 저장
        now = time.strftime('%y%m%d_%H%M%S', time.localtime(time.time()))
        save_dir = os.path.join('checkpoints', now)
        os.makedirs(save_dir, exist_ok=True)
        dataset_name = trainset.__str__().split()[1]
        torch.save(model.state_dict(), os.path.join(save_dir, 'unet_{}_{}.pth'.format(dataset_name, epoch)))

    writer.close()
