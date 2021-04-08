# -*- coding: utf-8 -*-
# Copyright (c) Youngwan Lee (ETRI) All Rights Reserved.

from detectron2.config import CfgNode as CN


def add_san_config(cfg):
    """
    Add config for VoVNet.
    """
    _C = cfg

    _C.MODEL.SAN = CN()
    _C.MODEL.SAN.CONV_BODY = "SAN19_pairwise"
    _C.MODEL.SAN.BASE_PATH = "../SAN/exp/imagenet/"
    _C.MODEL.SAN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]