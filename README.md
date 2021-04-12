# Mask-RCNN-SAN-Backbone

This project is based to use Self Attention Network (SAN) as backbone for feature extraction to Mask R-CNN Instance Segmentation Architecture. [SAN source paper](https://hszhao.github.io/papers/cvpr20_san.pdf) | [SAN source code](https://github.com/hszhao/SAN)

# Installation Guide

Please start installing Detectron 2 following [INSTALL.md.](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)

# COCO Dataset (2017)

Download COCO Dataset (2017 version)

Prepare for coco dataset following this [instructions](https://github.com/facebookresearch/detectron2/tree/master/datasets).

# Installation Guide

After installing Detectron2 and Download COCO Dataset Follow these steps:

Install Pytorch 1.7.1 & dependencies

pip install torch==1.7.1 torchvision==0.8.2 torchaudio=0.7.2

Install Cupy

pip install cupy-cuda110==7.8.0

Create $EXPORT local variable to set the path of the dataset

export DETECTRON2_DATASETS= {path/to/COCODataset/}

# Training

ImageNet Pretrained Models are already loaded at san-backbone. Selecting the config file will already load pre-trained models.

The authors of SAN model provide backbone weights pretrained on ImageNet-1k dataset.
    
    SAN-10 patchwise
    SAN-10 pairwise
    SAN-15 patchwise
    SAN-15 pairwise
    SAN-19 patchwise
    SAN-19 pairwise

To train a model, run

python /path/to/detectron2/projects/VoVNet/train_net.py --config-file projects/VoVNet/configs/<config.yaml>

For example, to launch end-to-end Mask R-CNN training with SAN-19 pairwise backbone on 1 GPUs, one should execute:

python train_net.py --config-file configs/mask_rcnn_SAN10_pairwise_FPN_3x.yaml --num-gpus 1

# Evaluation

Model evaluation can be done similarly:

python train_net.py --config-file configs/mask_rcnn_SAN10_pairwise_FPN_3x.yaml --eval-only MODEL.WEIGHTS <model.pth>
