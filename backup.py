import glob
import os
import shutil


def move_directory(src: str, dst):
    try:
        shutil.move(src, dst)
    except FileNotFoundError as e:
        print(e)


# Set backup directory
backup_dir = input('Directory name: ')
if backup_dir == '':
    backup_dir = 'rename'
backup_dir = os.path.join('backup', backup_dir)
try:
    os.makedirs(backup_dir)
except FileExistsError as e:
    print(e)

# Backup
move_directory('demo', backup_dir)
move_directory('feature_maps', backup_dir)
move_directory('runs', backup_dir)
move_directory('weights', backup_dir)
for file in glob.glob('result/*'):
    shutil.move(file, backup_dir)
