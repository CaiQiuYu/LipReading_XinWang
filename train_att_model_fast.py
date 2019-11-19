import random
import torch
import os
import pickle
from model.LipAttentionModel import LipModel
from tqdm import tqdm
from utils import cut_data, get_train_data, AverageMeter


def train():
    print("=====>> Select GPU")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # 需要更改
    random.seed(3020)
    val_number = 1200

    print("=====>> Parameter")
    train_path = 'data_pkl/data_train.pkl'

    batch_size = 16
    epochs = 10
    device = 'cuda:0'
    lr = 0.0005

    model = LipModel()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("======>> load data")
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
        label_data = pickle.load(f)
    print("======>> cut data to get val data")
    train_data, label_data, val_data, val_label = cut_data(train_data, label_data, val_number)
    print("======>> Generate torch train data")
    train_data, label_data = get_train_data(train_data, label_data, batch_size)
    val_data, val_label = get_train_data(val_data, val_label, batch_size)

    print("=======>> Start Training Model")
    best_acc = 0
    pred_re = []
    true_re = []
    for epoch in range(epochs):
        avg_loss = AverageMeter()
        # random batch
        data_lists = list(range(len(train_data)))
        random.shuffle(data_lists)
        model.train()
        for step, ids in tqdm(enumerate(data_lists)):
            inputs = train_data[ids].to(device)
            labels = label_data[ids].to(device)

            possibility, loss = model(inputs, labels)
            possibility = torch.argmax(possibility, dim=-1)
            loss = loss.mean()
            pred_re.append(possibility)
            true_re.append(labels)
            avg_loss.update(loss)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            step += 1
        print("Number of Epoch:" + str(epoch), end='\t')
        print("Current Avg loss:{0:.6f}".format(avg_loss.avg))
        pred_re = []
        true_re = []
        print("=====> Start Val")
        model.eval()
        acc = 0
        count = 0
        avg_loss_val = AverageMeter()
        with torch.no_grad():
            for ids in tqdm(range(len(val_data))):
                inputs = val_data[ids].to(device)
                labels = val_label[ids].to(device)
                possibility, loss = model(inputs, labels)
                avg_loss_val.update(loss.mean().item())
                count += inputs.size(0)
                acc += torch.sum(torch.eq(torch.argmax(possibility, dim=-1), labels)).item()
        acc = acc / count
        print("Current Val Loss: {0:.6f}  Accuracy: {1:.6f}".format(avg_loss_val.avg, acc))

        if acc >= best_acc:
            torch.save(model.state_dict(), "weights/attention/attention_net_{}.pt".format(epoch))
            best_acc = acc
            print("Saved Model, epoch = {0}".format(epoch))


if __name__ == '__main__':
    train()
