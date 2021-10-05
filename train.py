import os

import torch.distributed
import torch.utils.data
import torch.utils.tensorboard
import tqdm

import datasets
import eval
import utils

if __name__ == '__main__':
    # Load cfg and create components builder
    cfg = utils.builder.load_cfg()
    builder = utils.builder.Builder(cfg)

    # Distributed Data-Parallel Training (DDP)
    ddp_enabled = cfg['ddp_enabled']
    if ddp_enabled:
        assert torch.distributed.is_nccl_available(), 'NCCL backend is not available.'
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        os.system('clear')
    else:
        local_rank = 0
        world_size = 0

    # Device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')

    # 1. Dataset
    trainset, trainloader = builder.build_dataset('train', ddp_enabled)
    _, valloader = builder.build_dataset('val', ddp_enabled)

    # 2. Model
    model = builder.build_model(trainset.num_classes).to(device)
    if ddp_enabled:
        model = torch.nn.parallel.DistributedDataParallel(model)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model_name = cfg['model']['name']
    amp_enabled = cfg['model']['amp_enabled']
    print(f'Activated model: {model_name} (rank{local_rank})')

    # 3. Loss function, optimizer, lr scheduler, scaler
    criterion = builder.build_criterion(trainset.ignore_index)
    optimizer = builder.build_optimizer(model)
    scheduler = builder.build_scheduler(optimizer, len(trainloader) * cfg[model_name]['epoch'])
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    if cfg[model_name]['aux_criterion'] is not None:
        aux_criterion = builder.build_aux_criterion(trainset.ignore_index)
    else:
        aux_criterion = None

    # Resume training at checkpoint
    if cfg['resume_training'] is not None:
        path = cfg['resume_training']
        if ddp_enabled:
            torch.distributed.barrier()
            checkpoint = torch.load(path, map_location={'cuda:0': f'cuda:{local_rank}'})
        else:
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
        print(f'Resume training. {path} (rank{local_rank})')
    else:
        start_epoch = 0
        prev_miou = 0.0
        prev_val_loss = 2 ** 32 - 1

    # 4. Tensorboard
    if local_rank == 0:
        writer = torch.utils.tensorboard.SummaryWriter(os.path.join('runs', model_name))
    else:
        writer = None

    # 5. Train and evaluate
    for epoch in tqdm.tqdm(range(start_epoch, cfg[model_name]['epoch']),
                           desc='Epoch', disable=False if local_rank == 0 else True):
        if utils.train_interupter.train_interupter():
            print('Train interrupt occurs.')
            break
        if ddp_enabled:
            trainloader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        model.train()

        for batch_idx, (images, targets) in enumerate(tqdm.tqdm(trainloader, desc='Batch', leave=False,
                                                                disable=False if local_rank == 0 else True)):
            iters = len(trainloader) * epoch + batch_idx
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                if aux_criterion is None:
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                else:
                    outputs, aux = model(images)
                    loss = criterion(outputs, targets)
                    aux_loss = aux_criterion(aux)
                    loss += 0.4 * aux_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if ddp_enabled:
                loss_list = [torch.zeros(1, device=device) for _ in range(world_size)]
                torch.distributed.all_gather_multigpu([loss_list], [loss])
                if writer is not None:
                    for i, rank_loss in enumerate(loss_list):
                        writer.add_scalar(f'loss/training (rank{i})', rank_loss.item(), iters)
                    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iters)
            else:
                writer.add_scalar(f'loss/training (rank{local_rank})', loss.item(), iters)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iters)

            scheduler.step()

        # Evaluate
        val_loss, _, miou, _ = eval.evaluate(model, valloader, criterion, trainset.num_classes,
                                             amp_enabled, ddp_enabled, device)
        if writer is not None:
            writer.add_scalar('loss/validation', val_loss, epoch)
            writer.add_scalar('metrics/mIoU', miou, epoch)

        # Write predicted segmentation map
        if writer is not None:
            images, targets = valloader.__iter__().__next__()
            images, targets = images[2:4].to(device), targets[2:4]
            with torch.no_grad():
                outputs = model(images)
                outputs = torch.argmax(outputs, dim=1)
            if epoch == 0:
                targets = datasets.utils.decode_segmap_to_color_image(targets, trainset.colors, trainset.num_classes,
                                                                      trainset.ignore_index, trainset.ignore_color)
                writer.add_images('eval/0Groundtruth', targets, epoch)
            outputs = datasets.utils.decode_segmap_to_color_image(outputs, trainset.colors, trainset.num_classes)
            writer.add_images('eval/1' + model_name, outputs, epoch)

        if local_rank == 0:
            # Save checkpoint
            os.makedirs('weights', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'epoch': epoch,
                'miou': miou,
                'val_loss': val_loss
            }, os.path.join('weights', f'{model_name}_checkpoint.pth'))

            # Save best mIoU model
            if miou > prev_miou:
                torch.save(model.state_dict(), os.path.join('weights', f'{model_name}_best_miou.pth'))
                prev_miou = miou

            # Save best val_loss model
            if val_loss < prev_val_loss:
                torch.save(model.state_dict(), os.path.join('weights', f'{model_name}_best_val_loss.pth'))
                prev_val_loss = val_loss
    if writer is not None:
        writer.close()
    if ddp_enabled:
        torch.distributed.destroy_process_group()
