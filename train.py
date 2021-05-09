import os

import torch.utils.data
import torch.utils.tensorboard
import tqdm

import eval
import utils

if __name__ == '__main__':
    # 0. Load cfg and create components builder
    cfg = utils.builder.load_cfg()
    builder = utils.builder.Builder(cfg)

    # 1. Dataset
    dataset_impl, _, trainloader = builder.build_dataset('train')
    _, _, valloader = builder.build_dataset('val')

    # 2. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = builder.build_model(dataset_impl.num_classes).to(device)
    model_name = cfg['model']['name']
    amp_enabled = cfg['model']['amp_enabled']
    print(f'Activated model: {model_name}')

    # 3. Loss function, optimizer, lr scheduler, scaler
    criterion = builder.build_criterion(dataset_impl.ignore_index)
    optimizer = builder.build_optimizer(model)
    scheduler = builder.build_scheduler(optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # 4. Tensorboard
    writer = torch.utils.tensorboard.SummaryWriter(os.path.join('runs', model_name))

    # 5. Train and evaluate
    log_loss = tqdm.tqdm(total=0, position=2, bar_format='{desc}', leave=False)
    prev_miou = 0.0
    prev_val_loss = 100
    for epoch in tqdm.tqdm(range(cfg[model_name]['epoch']), desc='Epoch'):
        if utils.train_interupter.train_interupter():
            print('Train interrupt occurs.')
            break
        model.train()

        for batch_idx, (images, targets) in enumerate(tqdm.tqdm(trainloader, desc='Train', leave=False)):
            iters = len(trainloader) * epoch + batch_idx
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model(images)
                loss = criterion(output, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # lr scheduler의 step을 진행
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iters)
            scheduler.step()

            # 손실값 출력
            log_loss.set_description_str(f'Loss: {loss.item():.4f}')

            # Tensorboard에 학습 과정 기록
            writer.add_scalar('Train loss', loss.item(), iters)

        # 모델 평가
        val_loss, _, miou, _ = eval.evaluate(model, valloader, criterion, dataset_impl.num_classes,
                                             amp_enabled, device)
        writer.add_scalar('Validation loss', val_loss, epoch)
        writer.add_scalar('mIoU', miou, epoch)

        # Best mIoU를 가진 모델을 저장
        os.makedirs('weights', exist_ok=True)
        if miou > prev_miou:
            torch.save(model.state_dict(), os.path.join('weights', f'{model_name}_best.pth'))
            prev_miou = miou

        # Best val_loss를 가진 모델을 저장
        if val_loss < prev_val_loss:
            torch.save(model.state_dict(), os.path.join('weights', f'{model_name}_val_best.pth'))
            prev_val_loss = val_loss
    writer.close()
