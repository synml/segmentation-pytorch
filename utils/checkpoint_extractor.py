import torch


if __name__ == '__main__':
    checkpoint_name = input('Enter checkpoint name >>> ')
    checkpoint_path = '../weights/' + checkpoint_name
    checkpoint = torch.load(checkpoint_path)

    # Extract pretrained weight
    torch.save(checkpoint['model_best_miou'], f'{checkpoint_path.split("_")[0]}_best_miou.pth')
    torch.save(checkpoint['model_best_val_loss'], f'{checkpoint_path.split("_")[0]}_best_val_loss.pth')
