import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable

from LipReadDataTrain import ReadData
from model.LipD3DModel import LipModel
from collections import OrderedDict
from utils import AverageMeter

train_image_file = os.path.join(os.path.abspath('.'), "data_crop/lip_train")
train_label_file = os.path.join(os.path.abspath('.'), "data_original/train_list.txt")
training_dataset = ReadData(train_image_file, train_label_file, seq_max_lens=24)
training_data_loader = DataLoader(training_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)

val_image_file = os.path.join(os.path.abspath('.'), "data_crop/lip_train")
train_label_file = os.path.join(os.path.abspath('.'), "data_original/val_list.txt")
val_dataset = ReadData(train_image_file, train_label_file, seq_max_lens=24)
val_data_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=False)

# GPU
device_ids = [0]
learning_rate = 0.0001

model = LipModel()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model = nn.DataParallel(model, device_ids=device_ids)
model = model.cuda()


resume = 'weights/attention/attention_net_epoch_1.pt'
# optionally resume from a checkpoint
if os.path.isfile(resume):
    print("=> loading checkpoint '{}'".format(resume))
    checkpoint = torch.load(resume)
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        namekey = 'module.' + k  # remove `module.`
        new_state_dict[namekey] = v
    # load params
    model.load_state_dict(new_state_dict)
else:
    print("=> no checkpoint found at '{}'".format(resume))

log_step = 100
best_acc = 0
for epoch in range(1, 100):
    model.train()
    avg_loss = AverageMeter()
    pred = []
    true_result = []
    flag = 0
    for sample_batched in tqdm(training_data_loader):
        input_data = Variable(sample_batched['volume']).to(torch.device("cuda"))
        labels = Variable(sample_batched['label'].squeeze(1)).to(torch.device("cuda"))
        length = Variable(sample_batched['length']).to(torch.device("cuda"))

        possibility, loss = model(input_data, labels)
        possibility = torch.argmax(possibility, dim=-1)
        loss = loss.mean()
        pred.append(possibility)
        true_result.append(labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        avg_loss.update(loss.item(), input_data.size(0))

    print("Number of Epoch:" + str(epoch), end='\t')
    print("Current Avg loss:{0:.6f}".format(avg_loss.avg))
    print("======>>> Start Val")
    model.eval()
    with torch.no_grad():
        col_label = []
        col_pre = []
        avg_loss_val = AverageMeter()
        for sample_batched in tqdm(val_data_loader):
            input_data = Variable(sample_batched['volume']).to(torch.device("cuda"))
            labels = Variable(sample_batched['label'].squeeze(1)).to(torch.device("cuda"))

            outputs, loss = model(input_data, labels)
            avg_loss_val.update(loss.item(), input_data.size(0))
            col_label.append(labels)
            col_pre.append(labels)

    acc = torch.mean(torch.eq(torch.cat(col_pre), torch.cat(col_label)).float()).item()
    print("Current Val Loss: {0:.6f}  Accuracy: {1:.6f}".format(avg_loss_val.avg, acc))
    if best_acc < acc:
        # save model
        torch.save(model.module.state_dict(), "weights/attention/attention_net_{}.pt".format(epoch))


