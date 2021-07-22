import csv
import os
import time

import torch.distributed
import torch.utils.data
import tqdm

import utils


def evaluate(model: torch.nn.Module,
             valloader: torch.utils.data.DataLoader,
             criterion: torch.nn.Module,
             num_classes: int,
             amp_enabled: bool,
             ddp_enabled: bool,
             device: torch.device):
    model.eval()

    if ddp_enabled:
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        local_rank = 0
        world_size = 0

    evaluator = utils.metrics.Evaluator(num_classes, device)
    inference_time = torch.zeros(1, device=device)
    val_loss = torch.zeros(1, device=device)
    for images, targets in tqdm.tqdm(valloader, desc='Eval', leave=False, disable=False if local_rank == 0 else True):
        images, targets = images.to(device), targets.to(device)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            start_time = time.time()
            with torch.no_grad():
                outputs = model(images)
            inference_time += time.time() - start_time

            val_loss += criterion(outputs, targets)

            # Make segmentation map
            outputs = torch.argmax(outputs, dim=1)

        # Update confusion matrix
        evaluator.update_matrix(targets, outputs)

    if ddp_enabled:
        val_loss_list = [val_loss]
        confusion_matrix_list = [evaluator.confusion_matrix]
        inference_time_list = [inference_time]
        torch.distributed.all_reduce_multigpu(val_loss_list, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce_multigpu(confusion_matrix_list, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce_multigpu(inference_time_list, op=torch.distributed.ReduceOp.SUM)

        val_loss = val_loss_list[0] / (len(valloader) * world_size)
        evaluator.confusion_matrix = confusion_matrix_list[0]
        iou, miou = evaluator.get_scores()
        inference_time = inference_time_list[0] / (len(valloader) * world_size)
        fps = 1 / inference_time
    else:
        # Calculate average validation loss for batches
        val_loss /= len(valloader)

        # Get evaluation metrics
        iou, miou = evaluator.get_scores()

        # Calculate inference time and fps (inference time unit: seconds)
        inference_time /= len(valloader)
        fps = 1 / inference_time

    return val_loss.item(), iou, miou.item(), fps.item()


if __name__ == '__main__':
    # Load cfg and create components builder
    cfg = utils.builder.load_cfg()
    builder = utils.builder.Builder(cfg)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Dataset
    valset, valloader = builder.build_dataset('val')

    # 2. Model
    model = builder.build_model(valset.num_classes, pretrained=True).to(device)
    model_name = cfg['model']['name']
    amp_enabled = cfg['model']['amp_enabled']
    print(f'Activated model: {model_name}')

    # 3. Loss function
    criterion = builder.build_criterion(valset.ignore_index)

    # Evaluate model
    val_loss, iou, miou, fps = evaluate(model, valloader, criterion, valset.num_classes, amp_enabled, False, device)

    # Save evaluation result as csv file
    os.makedirs('result', exist_ok=True)
    class_names = valset.class_names
    with open(os.path.join('result', f'{model_name}.csv'), mode='w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerow(['Class Number', 'Class Name', 'IoU'])

        for class_num, iou_value in enumerate(iou):
            writer.writerow([class_num, class_names[class_num], iou_value.item()])
        writer.writerow(['mIoU', miou, ' '])
        writer.writerow(['Validation loss', val_loss, ' '])
        writer.writerow(['FPS', fps, ' '])
    print('Saved evaluation result.')
