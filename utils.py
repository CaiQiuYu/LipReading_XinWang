from tqdm import tqdm
import cv2
import glob
import numpy as np
import random
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def read_image(img_path):
    img_paths = sorted(glob.glob(img_path + '/*.png'))

    data = []
    for img_name in tqdm(img_paths):
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32)
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_CUBIC)
        img -= np.mean(img)
        img /= np.std(img)
        data.append(img)
    return np.array(data)


def cut_data(data_list, label_list, val_number=1000):
    train_data = []
    train_label = []
    val_data = []
    val_label = []
    val_idx = random.sample(range(len(data_list)), val_number)
    for i in tqdm(range(len(data_list))):
        if i not in val_idx:
            train_data.append(data_list[i])
            train_label.append(label_list[i])
        else:
            val_data.append(data_list[i])
            val_label.append(label_list[i])
    return train_data, train_label, val_data, val_label


def padding_data(data_batch):
    data = []
    time_steps = [a.shape[0] for a in data_batch]
    time_step = max(time_steps)
    for i, array in enumerate(data_batch):
        if array.shape[0] != time_step:
            t, h, w = array.shape
            pad_arr = np.zeros((time_step-t, h, w), dtype=np.float32)
            data_batch[i] = np.vstack((array, pad_arr))
        data.append(data_batch[i])
    return torch.tensor(data).unsqueeze(1)


def get_train_data(data_list, label_list, batch_size, test_data=False):
    data_train = []
    data_label = []
    number_data = len(data_list)
    number_batch = number_data // batch_size if number_data % batch_size == 0 else number_data // batch_size + 1

    batch_list = list(range(number_batch))
    random.shuffle(batch_list)

    for i in tqdm(batch_list):
        start = i * batch_size
        end = (i+1) * batch_size if (i+1) * batch_size < number_data else number_data

        data_train.append(padding_data(data_list[start:end]))
        if test_data:
            data_label.append(label_list[start:end])
        else:
            data_label.append(torch.tensor(label_list[start:end]))

    return data_train, data_label
