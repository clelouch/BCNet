import torch
import torch.nn.functional as F
from torch import nn


def upsample(tensor, size):
    return F.interpolate(tensor, size, mode='bilinear', align_corners=True)


def norm_layer(channel, norm_name='bn', _3d=False):
    if norm_name == 'bn':
        return nn.BatchNorm2d(channel) if not _3d else nn.BatchNorm3d(channel)
    elif norm_name == 'gn':
        return nn.GroupNorm(min(32, channel // 4), channel)


class RFB(nn.Module):
    """ receptive field block """

    def __init__(self, in_channel, out_channel=256):
        super(RFB, self).__init__()
        self.relu = nn.ReLU()
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3)  # 当kernel=3，如果dilation=padding则shape不变
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = nn.Conv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = torch.cat((x0, x1, x2, x3), 1)
        x_cat = self.conv_cat(x_cat)

        out = self.relu(x_cat + self.conv_res(x))
        return out


class GM(nn.Module):
    def __init__(self, channel):
        super(GM, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out1 = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out1 = out1.view(m_batchsize, C, height, width)

        return F.relu(x + out1)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=1):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class HFEM(nn.Module):
    # set dilation rate = 1
    def __init__(self, in_channel, out_channel):
        super(HFEM, self).__init__()
        self.in_C = in_channel
        self.temp_C = out_channel

        # first branch
        self.head_branch1 = nn.Sequential(
            nn.Conv2d(self.in_C, self.temp_C, 1, bias=False),
            norm_layer(self.temp_C),
            nn.ReLU(True)
        )
        self.conv1_branch1 = nn.Sequential(
            nn.Conv2d(self.temp_C, self.temp_C, 3, 1, 1, bias=False),
            norm_layer(self.temp_C),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )
        self.conv2_branch1 = nn.Sequential(
            nn.Conv2d(self.temp_C, self.temp_C, 3, 1, 1, bias=False),
            norm_layer(self.temp_C),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )
        self.conv3_branch1 = nn.Sequential(
            nn.Conv2d(self.temp_C, self.temp_C, 3, 1, 1, bias=False),
            norm_layer(self.temp_C),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )
        self.tail_branch1 = nn.Conv2d(self.temp_C, self.temp_C, 3, 1, padding=2, dilation=2, bias=False)

        # second branch
        self.head_branch2 = nn.Sequential(
            nn.Conv2d(self.in_C, self.temp_C, 1, bias=False),
            norm_layer(self.temp_C),
            nn.ReLU(True)
        )
        self.conv1_branch2 = nn.Sequential(
            nn.Conv2d(self.temp_C, self.temp_C, 3, 1, 1, bias=False),
            norm_layer(self.temp_C),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )
        self.conv2_branch2 = nn.Sequential(
            nn.Conv2d(self.temp_C, self.temp_C, 3, 1, 1, bias=False),
            norm_layer(self.temp_C),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )
        self.tail_branch2 = nn.Conv2d(self.temp_C, self.temp_C, 3, 1, padding=2, dilation=2, bias=False)

        # third branch
        self.head_branch3 = nn.Sequential(
            nn.Conv2d(self.in_C, self.temp_C, 1, bias=False),
            norm_layer(self.temp_C),
            nn.ReLU(True)
        )
        self.conv1_branch3 = nn.Sequential(
            nn.Conv2d(self.temp_C, self.temp_C, 3, 1, 1, bias=False),
            norm_layer(self.temp_C),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )
        self.tail_branch3 = nn.Conv2d(self.temp_C, self.temp_C, 3, 1, padding=2, dilation=2, bias=False)

        # forth branch
        self.head_branch4 = nn.Sequential(
            nn.Conv2d(self.in_C, self.temp_C, 1, bias=False),
            norm_layer(self.temp_C),
            nn.ReLU(True)
        )
        self.tail_branch4 = nn.Conv2d(self.temp_C, self.temp_C, 3, 1, padding=2, dilation=2, bias=False)

        # convs for fusion
        self.fusion_conv1 = nn.Sequential(
            nn.Conv2d(self.temp_C * 2, self.temp_C, 3, 1, 1, bias=False),
            norm_layer(self.temp_C),
            nn.ReLU(True)
        )
        self.fusion_conv2 = nn.Sequential(
            nn.Conv2d(self.temp_C * 2, self.temp_C, 3, 1, 1, bias=False),
            norm_layer(self.temp_C),
            nn.ReLU(True)
        )
        self.fusion_conv3 = nn.Sequential(
            nn.Conv2d(self.temp_C * 2, self.temp_C, 3, 1, 1, bias=False),  # output channel = temp_C
            norm_layer(self.temp_C),
            nn.ReLU(True)
        )

        # channel attention
        self.ca = ChannelAttention(out_channel)

    def forward(self, x):
        x_branch1_0 = self.head_branch1(x)
        x_branch1_1 = self.conv1_branch1(x_branch1_0)
        x_branch1_2 = self.conv2_branch1(x_branch1_1)
        x_branch1_3 = self.conv3_branch1(x_branch1_2)
        x_branch1_tail = self.tail_branch1(x_branch1_3)

        x_branch2_0 = self.head_branch2(x)
        x_branch2_0 = torch.cat([x_branch2_0,
                                 upsample(x_branch1_tail, x_branch2_0.shape[2:])], dim=1)
        x_branch2_0 = self.fusion_conv1(x_branch2_0)
        x_branch2_1 = self.conv1_branch2(x_branch2_0)
        x_branch2_2 = self.conv2_branch2(x_branch2_1)
        x_branch2_tail = self.tail_branch2(x_branch2_2)

        x_branch3_0 = self.head_branch3(x)
        x_branch3_0 = torch.cat([x_branch3_0,
                                 upsample(x_branch2_tail, x_branch3_0.shape[2:])], dim=1)
        x_branch3_0 = self.fusion_conv2(x_branch3_0)
        x_branch3_1 = self.conv1_branch3(x_branch3_0)
        x_branch3_tail = self.tail_branch3(x_branch3_1)

        x_branch4_0 = self.head_branch4(x)
        x_branch4_0 = torch.cat([x_branch4_0,
                                 upsample(x_branch3_tail, x_branch4_0.shape[2:])], dim=1)
        x_branch4_0 = self.fusion_conv3(x_branch4_0)
        x_branch4_tail = self.tail_branch4(x_branch4_0)

        # x_output = torch.cat([upsample(x_branch1_tail, x_branch4_tail.shape[2:]),
        #                       upsample(x_branch2_tail, x_branch4_tail.shape[2:]),
        #                       upsample(x_branch3_tail, x_branch4_tail.shape[2:]),
        #                       x_branch4_tail], dim=1)
        # x_output = self.fusion_cat(x_output)
        x_output = self.ca(x_branch4_tail) * x_branch4_tail
        return x_output


class Split(nn.Module):
    def __init__(self, channel=64):
        super(Split, self).__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            norm_layer(channel, _3d=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, channel, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            norm_layer(channel, _3d=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, channel, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            norm_layer(channel, _3d=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, channel, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            norm_layer(channel, _3d=True),
            nn.ReLU(inplace=True),
        )
        self.edge_pred = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, 3, 1, 1)
        )
        self.region_pred = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, 3, 1, 1)
        )

    def forward(self, x):
        edge = self.edge_conv(x)
        x = x.unsqueeze(2)
        edge = edge.unsqueeze(2)
        x_cat = torch.cat([x, edge], dim=2)
        x_cat_out = self.fusion(x_cat)

        x, edge = x_cat_out[:, :, 0, :, :], x_cat_out[:, :, 1, :, :]

        x_pred = self.region_pred(x)
        edge_pred = self.edge_pred(edge)
        return x, x_pred, edge, edge_pred


class SplitFusion(nn.Module):
    def __init__(self, channel=64):
        super(SplitFusion, self).__init__()
        self.reverse_conv = nn.Sequential(
            nn.Conv2d(channel + 1, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(True),
        )
        self.add_conv = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(True),
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(channel + 1, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(True),
        )
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            norm_layer(channel, _3d=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, channel, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            norm_layer(channel, _3d=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, channel, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            norm_layer(channel, _3d=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, channel, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            norm_layer(channel, _3d=True),
            nn.ReLU(inplace=True),
        )
        self.edge_pred = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, 3, 1, 1)
        )
        self.region_pred = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, 3, 1, 1)
        )

    def forward(self, x, rf, rp, bf, bp):
        rp = 1 - torch.sigmoid(rp)
        rp = upsample(rp, x.shape[2:])
        rf = upsample(rf, x.shape[2:])
        bp = upsample(bp, x.shape[2:])

        x = self.reverse_conv(torch.cat([x, rp], dim=1))
        x = self.add_conv(x + rf)

        edge = self.edge_conv(x)
        edge = self.cat_conv(torch.cat([edge, bp], dim=1))
        x = x.unsqueeze(2)
        edge = edge.unsqueeze(2)
        x_cat = torch.cat([x, edge], dim=2)
        x_cat_out = self.fusion(x_cat)

        x, edge = x_cat_out[:, :, 0, :, :], x_cat_out[:, :, 1, :, :]

        x_pred = self.region_pred(x)
        edge_pred = self.edge_pred(edge)
        return x, x_pred, edge, edge_pred
