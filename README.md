# [WACV'24] ODM3D: Alleviating Foreground Sparsity for Semi-Supervised Monocular 3D Object Detection

## Introduction
This repository contains the PyTorch implementation of [ODM3D](https://arxiv.org/abs/2310.18620) based on the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) codebase.


## KITTI Validation Results
|        Methods         |  3D Car Easy@R40  |  3D Car Mod@R40  |  3D Car Hard@R40  | BEV Car Easy@R40 | BEV Car Mod@R40 | BEV Car Hard@R40 |
|:----------------------:|:-----------------:|:----------------:|:-----------------:|:----------------:|:---------------:|:----------------:|
|     CMKD (ECCV'22)     |       30.20       |      21.50       |       19.40       |        -         |        -        |        -         |
| Mix-Teaching (CSVT'23) |       29.74       |      22.27       |       19.04       |      37.45       |      28.99      |      25.31       |
|     LPCG (ECCV'22)     |       31.15       |      23.42       |       20.60       |        -         |        -        |        -         |
|    ODM3D (WACV'24)     |     **35.76**     |    **24.30**     |     **20.66**     |    **45.04**     |    **31.35**    |    **27.45**     |

We report and provide a better performing model after some tuning. It has higher KITTI validation results that the ones reported in the paper with shorter training time.

## Checkpoints

|  Method   | Car Easy@R40 | Car Mod@R40 | Car Hard@R40 |                                         Student Model                                          | Teacher Model |
|:---------:|:------------:|:-----------:|:------------:|:----------------------------------------------------------------------------------------------:|:---:|
| ODM3D-R50 |    35.76     |    24.30    |    20.66     | [model](https://drive.google.com/file/d/1pqn0cJrRi6O4s17dB3BFjqlmAguvNgtb/view?usp=drive_link) | [model](https://drive.google.com/file/d/1NYlaQnS79dAsYSW85JR7NiHu2owc7-rc/view?usp=drive_link) |



## Installation

To-do. 
You may follow the guide provided in [CMKD](https://github.com/Cc-Hy/CMKD/blob/main/docs/INSTALL.md)

## Getting Started

To-do.
You may follow the guide provided in [CMKD](https://github.com/Cc-Hy/CMKD/blob/main/docs/GETTING_STARTED.md). Make sure you use [this](https://drive.google.com/file/d/1YxG2Yb1OhlscahsdWrwymY1yFcsOTaqN/view?usp=drive_link) training data file instead of the originally one, since the former contains pseudo-labels for unlabelled scenes.


## Citation

If you find our papers helpful, you may cite it as:
```
@inproceedings{odm3d,
author = {Weijia Zhang and Dongnan Liu and Chao Ma and Weidong Cai},
title = {Alleviating Foreground Sparsity for Semi-Supervised Monocular 3D Object Detection},
booktitle = {WACV},
year = {2024}
}
```