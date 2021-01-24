import glob
import os
import shutil


def move_directory(src: str, dst: str):
    if os.path.exists(src):
        shutil.move(src, dst)
    else:
        print('FileNotFound: ' + src)


def move_files(src: str, dst: str):
    if os.path.exists(src):
        for file in glob.glob(src + '/*'):
            shutil.move(file, dst)
        os.rmdir(src)
    else:
        print('FileNotFound: ' + src)


# Set backup directory
backup_dir = input('Directory name: ')
if backup_dir == '':
    backup_dir = 'rename'
backup_dir = os.path.join('backup', backup_dir)
os.makedirs(backup_dir)

# Backup
move_directory('demo', backup_dir)
move_directory('feature_maps', backup_dir)
move_directory('runs', backup_dir)
move_files('result', backup_dir)
move_files('weights', backup_dir)
