# Cross-Domain Adaptive Clustering for Semi-Supervised Domain Adaptation

This is a Pytorch implementation of "Cross-Domain Adaptive Clustering for Semi-Supervised Domain Adaptation" accepted by CVPR2021.
More details of this work can be found in our paper: [[Arxiv]](https://arxiv.org/abs/2104.09415) or [[OpenAccess]](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Cross-Domain_Adaptive_Clustering_for_Semi-Supervised_Domain_Adaptation_CVPR_2021_paper.html).

Our code is based on [SSDA_MME](https://github.com/VisionLearningGroup/SSDA_MME) implementation.

Note:
    The code will be gradually updated and completed as soon as possible.
## Install

`pip install -r requirements.txt`

The code is written in Python 3.8.5, but should work for other version with some modifications.


## Data preparation

Refer to [SSDA_MME](https://github.com/VisionLearningGroup/SSDA_MME).

## Training
To run training on DomainNet in the 3-shot scenario using resnet34,

`python main.py --dataset multi --source real --target sketch --net resnet34 --num 3 --lr_f 1.0 --multi 0.1`


### Reference
If you consider using this code or its derivatives, please consider citing:

```
@InProceedings{li2021cross,
    author    = {Li, Jichang and Li, Guanbin and Shi, Yemin and Yu, Yizhou},
    title     = {Cross-Domain Adaptive Clustering for Semi-Supervised Domain Adaptation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {2505-2514}
}
```
