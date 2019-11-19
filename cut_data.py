import random
import os
import glob


uniform_get = True
val_number = 1000
val_per_number = 3

train_path = 'data_crop/lip_train'
data_dict = {}
data_index = {}
with open('data_original/lip_train.txt', 'r', encoding='utf-8') as fr:
    for line in fr:
        word = line.strip().split('\t')
        new_path = train_path + '/' + word[0]
        image_list = sorted(glob.glob(new_path + '/*.*'), key=lambda x: int(x.split('\\')[-1][:-4]))
        length = len(image_list)
        data_index[word[0]] = word[1]
        if length == 0:
            print(word[0])
            continue
        if word[1] in data_dict:
            data_dict[word[1]].append(word[0])
        else:
            data_dict[word[1]] = [word[0]]


if uniform_get:
    file_names = sorted(os.listdir(train_path))
    train_data = []
    val_data = []
    for key in data_dict.keys():
        name_list = data_dict[key]
        val = random.sample(name_list, val_per_number)
        val_data.extend(val)
        for va in val:
            name_list.remove(va)
        train_data.extend(name_list)

    print(len(val_data))
    print(len(train_data))

    with open('data_original/train_list.txt', 'w', encoding='utf-8') as fw:
        for da in train_data:
            fw.write(da + '\t')
            fw.write(data_index[da] + '\n')

    with open('data_original/val_list.txt', 'w', encoding='utf-8') as fw:
        for da in val_data:
            fw.write(da + '\t')
            fw.write(data_index[da] + '\n')
else:
    all_keys = list(data_index.keys())
    val_data = random.sample(all_keys, val_number)
    train_data = []
    for k in all_keys:
        if k in val_data:
            continue
        else:
            train_data.append(k)

    with open('data_original/train_list.txt', 'w', encoding='utf-8') as fw:
        for da in train_data:
            fw.write(da + '\t')
            fw.write(data_index[da] + '\n')

    with open('data_original/val_list.txt', 'w', encoding='utf-8') as fw:
        for da in val_data:
            fw.write(da + '\t')
            fw.write(data_index[da] + '\n')