import os
import pandas as pd
import numpy as np
import cv2
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class ReadData(Dataset):

    def __init__(self, image_root, label_root, seq_max_lens, train_opt=True):
        self.seq_max_lens = seq_max_lens
        self.data = []
        self.data_root = image_root
        self.train_opt = train_opt
        with open(label_root, 'r', encoding='utf8') as f:
            lines = f.readlines()
            lines = [line.strip().split('\t') for line in lines]
            self.dictionary = sorted(np.unique([line[1] for line in lines])) 
            pic_path = [image_root + '/' + line[0] for line in lines]
            self.lengths = [len(os.listdir(path)) for path in pic_path]
            
            save_dict = pd.DataFrame(self.dictionary, columns=['dict'])
            save_dict.to_csv('./dictionary/dictionary.csv', encoding='utf8', index=None)  # save dict

            self.data = [(line[0], self.dictionary.index(line[1]), len(line[1]), line[1], length) for line, length in zip(lines, self.lengths)]
            # self.data_original = list(filter(lambda sample: sample[-1] <= self.seq_max_lens, self.data_original))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (path, label, label_length, label_name, pic_nums) = self.data[idx]
        tmp_path = path
        path = os.path.join(self.data_root, path)
        samples = np.round(np.linspace(0, pic_nums - 1, self.seq_max_lens))
        files = [os.path.join(path, ('{}' + '.png').format(int(i))) for i in samples]
        files = filter(lambda path: os.path.exists(path), files)
        frames = [cv2.imread(file) for file in files]
        frames_ = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in frames]       
        length = len(frames_)
        channels = 1
        picture_h_w = 112
        vlm = torch.zeros((channels, self.seq_max_lens, picture_h_w, picture_h_w))
        if self.train_opt:
            for i in range(len(frames_)):
                num = random.randint(0, 9)
                if num >= 5:
                    p = 1
                else:
                    p = 0
                result = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Grayscale(),
                    transforms.Resize((int(picture_h_w*1.2), int(picture_h_w*1.2))),
                    # transforms.CenterCrop((int(picture_h_w*1.2), int(picture_h_w*1.2))),
                    transforms.RandomCrop((picture_h_w, picture_h_w)),
                    transforms.RandomHorizontalFlip(p),
                    transforms.ToTensor(),
                    transforms.Normalize([0], [1])
                ])(frames_[i])
                vlm[:, i, :, :] = result
        else:
            for i in range(len(frames_)):
                result = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Grayscale(),
                    transforms.Resize((int(picture_h_w*1.2), int(picture_h_w*1.2))),
                    transforms.CenterCrop((int(picture_h_w), int(picture_h_w))),
                    # transforms.RandomCrop((picture_h_w, picture_h_w)),
                    transforms.ToTensor(),
                    transforms.Normalize([0], [1])
                ])(frames_[i])
                vlm[:, i, :, :] = result

        return {'volume': vlm, 'label': torch.LongTensor([label]),
                'word': label_name, 'length': length, 'pre_length': self.seq_max_lens,
                'label_length': label_length, 'key': tmp_path}


if __name__ == '__main__':
    train_image_file = os.path.join(os.path.abspath('.'), "data_original/lip_train")
    train_label_file = os.path.join(os.path.abspath('.'), "data_original/train_list.txt")
    training_dataset = ReadData(train_image_file, train_label_file, seq_max_lens=24)
    training_data_loader = DataLoader(training_dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=True)

    for i, sample_batched in enumerate(training_data_loader):
        print(sample_batched['true_length'])