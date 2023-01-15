import torch
from torch import nn
from modules import Split, SplitFusion, upsample
from torchvision import models
from ResNet import ResNet50
from Res2Net import Res2Net50
from modules import HFEM as EnhanceBlock


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config

        if config.backbone == 'resnet':
            self.resnet = ResNet50()
        elif config.backbone == 'res2net':
            self.resnet = Res2Net50()

        self.enhance5 = EnhanceBlock(2048, 64)
        self.enhance4 = EnhanceBlock(1024, 64)
        self.enhance3 = EnhanceBlock(512, 64)
        self.enhance2 = EnhanceBlock(256, 64)

        self.sp = Split()
        self.sf1 = SplitFusion()
        self.sf2 = SplitFusion()
        self.sf3 = SplitFusion()

        self._initialize_weight()

    def forward(self, x):
        target_size = x.shape[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        x4 = self.enhance5(x4)
        x3 = self.enhance4(x3)
        x2 = self.enhance3(x2)
        x1 = self.enhance2(x1)

        x4_rf, x4_rp, x4_bf, x4_bp = self.sp(x4)
        x3_rf, x3_rp, x3_bf, x3_bp = self.sf1(x3, x4_rf, x4_rp, x4_bf, x4_bp)
        x2_rf, x2_rp, x2_bf, x2_bp = self.sf2(x2, x3_rf, x3_rp, x3_bf, x3_bp)
        x1_rf, x1_rp, x1_bf, x1_bp = self.sf3(x1, x2_rf, x2_rp, x2_bf, x2_bp)

        return [torch.sigmoid(upsample(x1_rp, target_size)), torch.sigmoid(upsample(x2_rp, target_size)),
                torch.sigmoid(upsample(x3_rp, target_size)), torch.sigmoid(upsample(x4_rp, target_size))], \
               [torch.sigmoid(upsample(x1_bp, target_size)), torch.sigmoid(upsample(x2_bp, target_size)),
                torch.sigmoid(upsample(x3_bp, target_size)), torch.sigmoid(upsample(x4_bp, target_size))]

    def _initialize_weight(self):
        if self.config.backbone == 'resnet':
            res50 = models.resnet50(pretrained=True)
            pretrained_dict = res50.state_dict()
        else:
            pretrained_dict = torch.load('./models/res2net50_v1b_26w_4s-3cf99910.pth')
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)


if __name__ == '__main__':
    from options import opt
    model = Net(opt)
    img = torch.randn((3, 3, 256, 256))
    out = model(img)