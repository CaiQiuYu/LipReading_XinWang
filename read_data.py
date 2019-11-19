import os
import pickle
from utils import read_image


train_path = 'data_crop/lip_train/'
test_path = 'data_crop/lip_test/'
label_path = 'data_original/lip_train.txt'
save_path = 'data_pkl/'
dictionary_path = 'dictionary/dictionary.txt'

print("=============> Generate dictionary list")
index2word = {}
word2label = {}
with open(label_path, 'r', encoding='utf-8') as fr:
    for lines in fr:
        words = lines.strip().split('\t')
        index2word[words[0]] = words[1]
        if words[1] not in word2label:
            word2label[words[1]] = len(word2label)

with open(dictionary_path, 'w', encoding='utf-8') as fw:
    for word in word2label.keys():
        fw.write(word + ',' + str(word2label[word]))
        fw.write('\n')

print("=============> Process Train Data")
file_dirs = os.listdir(train_path)
file_count = {}
for fi in file_dirs:
    num = len(os.listdir(os.path.join(train_path, fi)))
    if num not in file_count:
        file_count[num] = [fi]
    else:
        file_count[num].append(fi)

train_data = []
train_label = []
data_keys = sorted(file_count.keys())
for k in data_keys:
    for name in file_count[k]:
        path = os.path.join(train_path, name)
        image = read_image(path)
        if image is not None:
            train_data.append(image)
            train_label.append(word2label[index2word[name]])

print("=============> Save Train Data")
with open(os.path.join(save_path, 'data_train.pkl'), 'wb') as f:
    pickle.dump(train_data, f)
    pickle.dump(train_label, f)

print("=============> Process Test Data")
file_dirs = os.listdir(test_path)
file_count = {}
for fi in file_dirs:
    num = len(os.listdir(os.path.join(test_path, fi)))
    if num not in file_count:
        file_count[num] = [fi]
    else:
        file_count[num].append(fi)

test_data = []
test_names = []
data_keys = sorted(file_count.keys())
for k in data_keys:
    for name in file_count[k]:
        path = os.path.join(test_path, name)
        image = read_image(path)
        if image is not None:
            test_data.append(image)
            test_names.append(name)

print("=============> Save Test Data")
with open(os.path.join(save_path, 'data_test.pkl'), 'wb') as f:
    pickle.dump(test_data, f)
    pickle.dump(test_names, f)
