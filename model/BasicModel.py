import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules import ResBlock


class LipModel(nn.Module):
    def __init__(self, num_class=313):
        super(LipModel, self).__init__()
        self.conv3d = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        self.bn = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool3d = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))
        self.res_channel = 64
        self.resnet = nn.Sequential(
            self.ResLayer(64, n_block=1, stride=1),
            self.ResLayer(128, n_block=2, stride=2),
            self.ResLayer(256, n_block=2, stride=2),
            self.ResLayer(512, n_block=2, stride=2)
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=1)
        self.res_linear = nn.Linear(self.res_channel, 256)
        self.linear_bn = nn.BatchNorm1d(256)
        self.lstm = nn.GRU(256, 256, num_layers=1, bidirectional=True)
        self.classfication = nn.Linear(2*256, num_class)

    def ResLayer(self, out_channel, n_block, stride):
        strides = [stride] + [1] * (n_block-1)
        layers = []
        for stride in strides:
            layers.append(ResBlock(self.res_channel, out_channel, stride))
            self.res_channel = out_channel
        return nn.Sequential(*layers)

    def forward(self, feature, label=None):
        # Get 3D feature
        output = self.conv3d(feature)
        output = self.bn(output)
        output = self.relu(output)
        output = self.pool3d(output)
        output = torch.transpose(output, 1, 2).contiguous()
        batch, step, channel, h, w = output.size()
        output = output.view(-1, channel, h, w)

        # Get 2D ResBet feature
        res_feature = self.resnet(output)
        res_feature = self.avg_pool(res_feature)
        res_feature = res_feature.view(batch * step, self.res_channel)
        res_feature = self.res_linear(res_feature)
        res_feature = self.linear_bn(res_feature)
        res_feature = res_feature.view(batch, step, -1)

        res_feature = res_feature.transpose(0, 1).contiguous()
        rnn_feature = self.lstm(res_feature)[0]
        rnn_feature = rnn_feature.transpose(0, 1)

        final_output = self.classfication(rnn_feature)

        # Get Scores
        scores = F.softmax(final_output, -1)
        scores = torch.sum(scores, dim=1)
        result = (scores,)

        # Calculate loss
        if label is not None:
            log_scores = torch.mean(-F.log_softmax(final_output, -1), dim=1)
            loss = log_scores.gather(dim=-1, index=label[:, None]).squeeze()
            result = (scores, loss)

        return result


if __name__ == '__main__':
    lip_net = LipModel()
    a = torch.randn(32, 1, 24, 112, 112)
    out = lip_net(a, None)
    print(out[0].size())