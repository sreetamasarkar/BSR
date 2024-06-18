![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)
[![arxiv](http://img.shields.io/badge/arxiv-2310.04562-yellow.svg)](https://arxiv.org/abs/2405.10951)
# Block Selective Reprogramming for On-device Training of Vision Transformers

## Overview
This is the Pytorch code for the paper Block Selective Reprogramming for On-device Training of Vision Transformers. This paper presents block selective reprogramming (BSR) in which we fine-tune only a fraction of total blocks of a pre-trained model and selectively drop tokens based on self-attention scores of the frozen layers. To show the efficacy of BSR, we present extensive evaluations on ViT-B and DeiT-S with five different datasets. Compared to the existing alternatives, our approach simultaneously reduces training memory by up to 1.4x and compute cost by up to 2x while maintaining similar accuracy. We also showcase results for Mixture-of-Expert (MoE) models, demonstrating the effectiveness of our approach in multitask learning scenarios.
<p align="center"> 
    <img src="system_diagram.svg">
</p> 


## Authors

- [Sreetama Sarkar](https://www.linkedin.com/in/sreetama-sarkar-332a13104/) (USC)
- [Souvik Kundu](https://www.linkedin.com/in/souvik-kundu-ph-d-64922b50/) (Intel)
- [Kai Zheng]() (USC)
- [Peter A. Beerel](https://sites.usc.edu/eessc/) (USC)
  
## Paper
[link to paper](https://arxiv.org/abs/2405.10951)

## Installation
The code is based on torch==1.13.1, torchvision==0.14.1 and timm==0.3.2

For running experiments on MoE models, follow the installation instructions in [M3ViT](https://github.com/VITA-Group/M3ViT/tree/main)

## Training
Model training using BSR can be performed using the scripts provided in exp_scripts/


## Acknowledgement
The partial code for this repo is taken from [BackRazor](https://github.com/VITA-Group/BackRazor_Neurips22), [MAE](https://github.com/facebookresearch/mae?tab=readme-ov-file), [M3ViT](https://github.com/VITA-Group/M3ViT/tree/main) and [EViT](https://github.com/youweiliang/evit)

## Citation
If you find this repo useful for your research, please consider citing the following work:
```
@InProceedings{sarkarECV24,
    author    = {Sarkar, Sreetama and Kundu, Souvik and Zheng, Kai and Beerel, Peter},
    title     = {Block Selective Reprogramming for On-device Training of Vision Transformers},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    year      = {2024}
}