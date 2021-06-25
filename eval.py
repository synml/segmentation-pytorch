import csv
import os
import time

import torch.utils.data
import tqdm

import utils


def evaluate(model, testloader, criterion, num_classes: int, amp_enabled: bool, device):
    model.eval()

    evaluator = utils.metrics.Evaluator(num_classes)
    val_loss = 0.0
    inference_time = 0.0
    for images, targets in tqdm.tqdm(testloader, desc='Eval', leave=False):
        images, targets = images.to(device), targets.to(device)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            start_time = time.time()
            with torch.no_grad():
                outputs = model(images)
            inference_time += time.time() - start_time

            val_loss += criterion(outputs, targets).item()

            # Make segmentation map
            outputs = torch.argmax(outputs, dim=1)

        # Update confusion matrix
        evaluator.update_matrix(targets, outputs)

    # Calculate average validation loss for batches
    val_loss /= len(testloader)

    # Get evaluation metrics
    iou, miou = evaluator.get_scores()

    # Calculate inference time and fps (inference time unit: seconds)
    inference_time /= len(testloader)
    fps = 1 / inference_time

    return val_loss, iou, miou, fps


if __name__ == '__main__':
    # 0. Load cfg and create components builder
    cfg = utils.builder.load_cfg()
    builder = utils.builder.Builder(cfg)

    # 1. Dataset
    valset, valloader = builder.build_dataset('val')

    # 2. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = builder.build_model(valset.num_classes, pretrained=True).to(device)
    model_name = cfg['model']['name']
    amp_enabled = cfg['model']['amp_enabled']
    print(f'Activated model: {model_name}')

    # 3. Loss function
    criterion = builder.build_criterion(valset.ignore_index)

    # Evaluate model
    val_loss, iou, miou, fps = evaluate(model, valloader, criterion, valset.num_classes, amp_enabled, device)

    # Save evaluation result as csv file
    os.makedirs('result', exist_ok=True)
    class_names = valset.class_names
    with open(os.path.join('result', f'{model_name}.csv'), mode='w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerow(['Class Number', 'Class Name', 'IoU'])

        for class_num, iou_value in enumerate(iou):
            writer.writerow([class_num, class_names[class_num], iou_value])
        writer.writerow(['mIoU', miou, ' '])
        writer.writerow(['Validation loss', val_loss, ' '])
        writer.writerow(['FPS', fps, ' '])
    print('Saved evaluation result.')
