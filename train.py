import os

import torch.utils.data
import torch.utils.tensorboard
import tqdm

import datasets
import eval
import utils

if __name__ == '__main__':
    # 0. Load cfg and create components builder
    cfg = utils.builder.load_cfg()
    builder = utils.builder.Builder(cfg)

    # 1. Dataset
    trainset, trainloader = builder.build_dataset('train')
    _, valloader = builder.build_dataset('val')

    # 2. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = builder.build_model(trainset.num_classes).to(device)
    model_name = cfg['model']['name']
    amp_enabled = cfg['model']['amp_enabled']
    print(f'Activated model: {model_name}')

    # 3. Loss function, optimizer, lr scheduler, scaler
    criterion = builder.build_criterion(trainset.ignore_index)
    optimizer = builder.build_optimizer(model)
    scheduler = builder.build_scheduler(optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # 4. Tensorboard
    writer = torch.utils.tensorboard.SummaryWriter(os.path.join('runs', model_name))

    # Resume training at checkpoint
    if cfg['resume_training'] is not None:
        path = cfg['resume_training']
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if cfg['fine_tuning_batchnorm']:
            model.freeze_bn()
        else:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        prev_miou = checkpoint['miou']
        prev_val_loss = checkpoint['val_loss']
        print(f'Resume training. {path}')
    else:
        start_epoch = 0
        prev_miou = 0.0
        prev_val_loss = 100

    # Train and evaluate
    log_loss = tqdm.tqdm(total=0, position=2, bar_format='{desc}', leave=False)
    for epoch in tqdm.tqdm(range(start_epoch, cfg[model_name]['epoch']), desc='Epoch'):
        if utils.train_interupter.train_interupter():
            print('Train interrupt occurs.')
            break
        model.train()

        for batch_idx, (images, targets) in enumerate(tqdm.tqdm(trainloader, desc='Train', leave=False)):
            iters = len(trainloader) * epoch + batch_idx
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            writer.add_scalar('loss/training', loss.item(), iters)
            log_loss.set_description_str(f'Loss: {loss.item():.4f}')

            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iters)
            scheduler.step()

        val_loss, _, miou, _ = eval.evaluate(model, valloader, criterion, trainset.num_classes, amp_enabled, device)
        writer.add_scalar('loss/validation', val_loss, epoch)
        writer.add_scalar('metrics/mIoU', miou, epoch)

        images, targets = valloader.__iter__().__next__()
        images, targets = images[:4].to(device), targets[:4]
        with torch.no_grad():
            outputs = model(images)
            outputs = torch.argmax(outputs, dim=1)
        writer.add_images('eval/0Groundtruth',
                          datasets.test.decode_segmap(targets, trainset.get_colormap(), trainset.num_classes,
                                                      trainset.ignore_index), epoch)
        writer.add_images('eval/1' + model_name,
                          datasets.test.decode_segmap(outputs.cpu(), trainset.get_colormap(), trainset.num_classes,
                                                      trainset.ignore_index), epoch)

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'epoch': epoch,
            'miou': miou,
            'val_loss': val_loss,
            'model_best_miou': None,
            'model_best_val_loss': None
        }

        # Save best mIoU model
        if miou > prev_miou:
            checkpoint['model_best_miou'] = model.state_dict()
            prev_miou = miou

        # Save best val_loss model
        if val_loss < prev_val_loss:
            checkpoint['model_best_val_loss'] = model.state_dict()
            prev_val_loss = val_loss

        # Save checkpoint
        os.makedirs('weights', exist_ok=True)
        torch.save(checkpoint, os.path.join('weights', f'{model_name}_checkpoint.pth'))
    writer.close()
