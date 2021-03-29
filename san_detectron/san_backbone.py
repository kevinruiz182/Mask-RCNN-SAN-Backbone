# Copyright (c) Youngwan Lee (ETRI) All Rights Reserved.
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.layers import FrozenBatchNorm2d, ShapeSpec, get_norm
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool

# SAN Libraries
import sys
sys.path.append('../SAN')
from SAN.lib.sa.modules import Subtraction, Subtraction2, Aggregation

__all__ = ["SAN", "build_san_backbone", "build_san_fpn_backbone"]

SAN10_pairwise = {
    'sa_type': 0,
    'layers': (2, 1, 2, 4, 1),
    'kernels': [3, 7, 7, 7, 7],
    "num_classes": 1000,
}

SAN10_patchwise = {
    'sa_type': 1,
    'layers': (2, 1, 2, 4, 1),
    'kernels': [3, 7, 7, 7, 7],
    "num_classes": 1000,
}

SAN15_pairwise = {
    'sa_type': 0,
    'layers': (3, 2, 3, 5, 2),
    'kernels': [3, 7, 7, 7, 7],
    "num_classes": 1000,
}

SAN15_patchwise = {
    'sa_type': 1,
    'layers': (3, 2, 3, 5, 2),
    'kernels': [3, 7, 7, 7, 7],
    "num_classes": 1000,
}

SAN19_pairwise = {
    'sa_type': 0,
    'layers': (3, 3, 4, 6, 3),
    'kernels': [3, 7, 7, 7, 7],
    "num_classes": 1000,
}

SAN19_patchwise = {
    'sa_type': 1,
    'layers': (3, 3, 4, 6, 3),
    'kernels': [3, 7, 7, 7, 7],
    "num_classes": 1000,
}

_SAN_PARAMS = {
    "SAN10_pairwise": SAN10_pairwise,
    "SAN10_patchwise": SAN10_patchwise,
    "SAN15_pairwise": SAN15_pairwise,
    "SAN15_patchwise": SAN15_patchwise,
    "SAN19_pairwise": SAN19_pairwise,
    "SAN19_patchwise": SAN19_patchwise,
}


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


class SAM(nn.Module):
    def __init__(self, sa_type, in_planes, rel_planes, out_planes, share_planes, kernel_size=3, stride=1, dilation=1):
        super(SAM, self).__init__()
        self.sa_type, self.kernel_size, self.stride = sa_type, kernel_size, stride
        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        if sa_type == 0:
            self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes + 2), nn.ReLU(inplace=True),
                                        nn.Conv2d(
                                            rel_planes + 2, rel_planes, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(rel_planes), nn.ReLU(
                                            inplace=True),
                                        nn.Conv2d(rel_planes, out_planes // share_planes, kernel_size=1))
            self.conv_p = nn.Conv2d(2, 2, kernel_size=1)
            self.subtraction = Subtraction(
                kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
            self.subtraction2 = Subtraction2(
                kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
            self.softmax = nn.Softmax(dim=-2)
        else:
            self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes * (pow(kernel_size, 2) + 1)), nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes * (pow(kernel_size, 2) + 1),
                                                  out_planes // share_planes, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(
                                            out_planes // share_planes), nn.ReLU(inplace=True),
                                        nn.Conv2d(out_planes // share_planes, pow(kernel_size, 2) * out_planes // share_planes, kernel_size=1))
            self.unfold_i = nn.Unfold(
                kernel_size=1, dilation=dilation, padding=0, stride=stride)
            self.unfold_j = nn.Unfold(
                kernel_size=kernel_size, dilation=dilation, padding=0, stride=stride)
            self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.aggregation = Aggregation(
            kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)

    def forward(self, x):
        x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)
        if self.sa_type == 0:  # pairwise
            p = self.conv_p(position(x.shape[2], x.shape[3], x.is_cuda))
            w = self.softmax(self.conv_w(torch.cat([self.subtraction2(
                x1, x2), self.subtraction(p).repeat(x.shape[0], 1, 1, 1)], 1)))
        else:  # patchwise
            if self.stride != 1:
                x1 = self.unfold_i(x1)
            x1 = x1.view(x.shape[0], -1, 1, x.shape[2]*x.shape[3])
            x2 = self.unfold_j(self.pad(x2)).view(
                x.shape[0], -1, 1, x1.shape[-1])
            w = self.conv_w(torch.cat([x1, x2], 1)).view(
                x.shape[0], -1, pow(self.kernel_size, 2), x1.shape[-1])
        x = self.aggregation(x3, w)
        return x


class Bottleneck(nn.Module):
    def __init__(self, sa_type, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, kernel_size=7, stride=1):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.sam = SAM(sa_type, in_planes, rel_planes,
                       mid_planes, share_planes, kernel_size, stride)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv = nn.Conv2d(mid_planes, out_planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(x))
        out = self.relu(self.bn2(self.sam(out)))
        out = self.conv(out)
        out += identity
        return out


class SAN(Backbone):
    def __init__(self, cfg, sa_type, block, layers, kernels, num_classes):
        super(SAN, self).__init__()

        self.stage_names = []
        #First Block
        dict1 = OrderedDict()
        c = 64
        conv_in, bn_in = conv1x1(3, c), nn.BatchNorm2d(c)
        conv0, bn0 = conv1x1(c, c), nn.BatchNorm2d(c)
        layer0 = self._make_layer(
            sa_type, block, c, layers[0], kernels[0])

        dict1['conv_in'] = conv_in
        dict1['bn_in'] = bn_in
        dict1['conv0'] = conv0
        dict1['bn0'] = bn0
        dict1['layer0'] = layer0

        self.add_module("stem", nn.Sequential(dict1))

        #Second Block
        dict2 = OrderedDict()
        c *= 4
        conv1, bn1 = conv1x1(c // 4, c), nn.BatchNorm2d(c)
        layer1 = self._make_layer(
            sa_type, block, c, layers[1], kernels[1])

        dict2['conv1'] = conv1
        dict2['bn1'] = bn1
        dict2['layer1'] = layer1

        self.stage_names.append("res2")
        self.add_module("res2", nn.Sequential(dict2))

        #Third Block
        dict3 = OrderedDict()
        c *= 2
        conv2, bn2 = conv1x1(c // 2, c), nn.BatchNorm2d(c)
        layer2 = self._make_layer(
            sa_type, block, c, layers[2], kernels[2])

        dict3['conv2'] = conv2
        dict3['bn2'] = bn2
        dict3['layer2'] = layer2

        self.stage_names.append("res3")
        self.add_module("res3", nn.Sequential(dict3))

        #Fourth Block
        dict4 = OrderedDict()
        c *= 2
        conv3, bn3 = conv1x1(c // 2, c), nn.BatchNorm2d(c)
        layer3 = self._make_layer(
            sa_type, block, c, layers[3], kernels[3])

        dict4['conv3'] = conv3
        dict4['bn3'] = bn3
        dict4['layer3'] = layer3

        self.stage_names.append("res4")
        self.add_module("res4", nn.Sequential(dict4))

        #Fifth Block
        dict5 = OrderedDict()
        c *= 2
        conv4, bn4 = conv1x1(c // 2, c), nn.BatchNorm2d(c)
        layer4 = self._make_layer(
            sa_type, block, c, layers[4], kernels[4])

        # self.relu = nn.ReLU(inplace=True)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(c, num_classes)

        dict5['conv4'] = conv4
        dict5['bn4'] = bn4
        dict5['layer4'] = layer4

        self.stage_names.append("res5")
        self.add_module("res5", nn.Sequential(dict5))


        # self._initialize_weights()
        # self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_AT)

    def _freeze_backbone(self, freeze_at):
        print("freeze_at:")
        print(freeze_at)
        for layer_index in range(freeze_at):
            for p in self.features[layer_index].parameters():
                p.requires_grad = False

    def _make_layer(self, sa_type, block, planes, blocks, kernel_size=7, stride=1):
        layers = []
        for _ in range(0, blocks):
            layers.append(block(sa_type, planes, planes // 16,
                                planes // 4, planes, 8, kernel_size, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = x[:, [2, 1, 0], : , :]
        outputs = {}
        x = self.stem(x)
        # if "stem" in self._out_features:
        outputs["stem"] = x
        for name in self.stage_names:
            print("name")
            print(name)
            x = getattr(self, name)(x)
            # if name in self._out_features:
            outputs[name] = x

        return outputs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** 0.5)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def san(cfg, sa_type, block, layers, kernels, num_classes):
    model = SAN(cfg, sa_type, block, layers, kernels, num_classes)
    # model = torch.nn.DataParallel(model.cuda())
    # checkpoint = torch.load("../SAN/exp/imagenet/san19_patchwise/model/model_best.pth", map_location= "cpu")
    # model.load_state_dict(checkpoint['state_dict'], strict=True)
    # toModules = [module for module in model.modules() if not isinstance(module, nn.Sequential)]
    # return toModules[1]
    return model

@BACKBONE_REGISTRY.register()
def build_san_backbone(cfg, input_shape):

    """
    Create a SAN instance from config.

    Returns:
        SAN: a :class:`SAN` instance.
    """

    out_features = cfg.MODEL.SAN.OUT_FEATURES

    out_feature_channels = {"res2": 256, "res3": 512,
                            "res4": 1024, "res5": 2048}
    out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}

    # Set SAN parameters according to config (_SAN_PARAMS)

    san_params = _SAN_PARAMS[cfg.MODEL.SAN.CONV_BODY]
    san_sa_type = san_params['sa_type']
    san_layers = san_params['layers']
    san_kernels = san_params['kernels']
    san_num_classes = san_params['num_classes']


    model = san(cfg,
                sa_type=san_sa_type,
                block=Bottleneck,
                layers=san_layers,
                kernels=san_kernels,
                num_classes=san_num_classes)


    model._out_features = out_features
    model._out_feature_channels = out_feature_channels
    model._out_feature_strides = out_feature_strides
    return model



@BACKBONE_REGISTRY.register()
def build_san_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_san_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone