import os
import shutil
base_path = "../data/food-101"
images_dir = os.path.join(base_path, 'images')
meta_dir = os.path.join(base_path, 'meta')

#output dirs
split_base = '../data/split'
train_dir = os.path.join(split_base, 'train')
val_dir = os.path.join(split_base, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

#load split file lists
with open(os.path.join(meta_dir, 'train.txt')) as f:
    train_files = [line.strip() for line in f]

with open(os.path.join(meta_dir, 'test.txt')) as f:
    val_files = [line.strip() for line in f]

def copy_files(file_list, target_root):
    for item in file_list:
        label, img = item.split('/')
        src = os.path.join(images_dir, label, img + '.jpg')
        dest_dir = os.path.join(target_root, label)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(dest_dir, img + '.jpg'))

print("Copying training images...")
copy_files(train_files, train_dir)
print("Copying validation images...")
copy_files(val_files, val_dir)
print("Done splitting dataset.")