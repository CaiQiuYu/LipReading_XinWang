import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from model.modules import _DenseBlock, _Transition, ResBlock


class LipModel(nn.Module):
    def __init__(self, num_class=313):
        super(LipModel, self).__init__()

        # 3D卷积
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)),
            ('norm0', nn.BatchNorm3d(64)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))),
        ]))

        block = _DenseBlock(num_layers=1, num_input_features=64,
                            bn_size=4, growth_rate=32, drop_rate=0.5)
        self.features.add_module('denseblock1', block)
        trans = _Transition(num_input_features=96, num_output_features=128)
        self.features.add_module('transition%d1', trans)
        self.features.add_module('norm', nn.BatchNorm3d(128))

        # ResNet
        self.res_channel = 128
        self.resnet = nn.Sequential(
            self.ResLayer(128, n_block=1, stride=1),
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
        output = self.features(feature)
        output = torch.transpose(output, 1, 2).contiguous()
        batch, step, channel, h, w = output.size()
        output = output.view(-1, channel, h, w)

        # Get 2D ResBet feature
        res_feature = self.resnet(output)
        res_feature = self.avg_pool(res_feature)
        res_feature = res_feature.view(batch*step, self.res_channel)
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
    print(lip_net)
    a = torch.randn(4, 1, 24, 112, 112)
    out = lip_net(a, None)
    print(out[0].size())