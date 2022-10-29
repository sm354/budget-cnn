import glob
import os
import PIL.Image as Image
import numpy as np

'''
creates numpy uint8 files : train_x, train_y, test_x, test_y

'''

train_x, train_y, test_x, test_y = [], [], [], []

train_folder = './tiny-imagenet-200/train/'
val_folder = './tiny-imagenet-200/val/'

label_name2id_dict = {}

for label_id, label_name in enumerate(os.listdir(train_folder)):

    label_name2id_dict[label_name] = label_id

    imgs_path = os.path.join(train_folder, label_name, 'images')
    for img_name in os.listdir(imgs_path):
        x = Image.open(os.path.join(imgs_path,img_name))
        x = x.resize((64,64))
        x = np.array(x)

        if len(x.shape)==3:
            train_x.append(x)
            train_y.append(label_id)
            # break

print(f"training data size {len(train_x)},{len(train_y)}")
train_x = np.array(train_x, dtype=np.uint8)
train_y = np.array(train_y, dtype=np.uint8)
np.save('./train_x.npy', train_x)
np.save('./train_y.npy', train_y)

val_dict = {}
with open('./tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
    for line in f.readlines():
        split_line = line.split('\t')
        val_dict[split_line[0]] = split_line[1]
        
paths = glob.glob('./tiny-imagenet-200/val/images/*') 

imgs_path = os.path.join(val_folder, 'images')
for img_name in os.listdir(imgs_path):
    x = Image.open(os.path.join(imgs_path,img_name))
    x = x.resize((64,64))
    x = np.array(x)

    if len(x.shape)==3:
        test_x.append(x)
        label_id = label_name2id_dict[val_dict[img_name]]
        test_y.append(label_id)

print(f"testing data size {len(test_x)},{len(test_y)}")
test_x = np.array(test_x, dtype=np.uint8)
test_y = np.array(test_y, dtype=np.uint8)
np.save('./test_x.npy', test_x)
np.save('./test_y.npy', test_y)