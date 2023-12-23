# Cross-Domain Adaptive Clustering for Semi-Supervised Domain Adaptation

This is a Pytorch implementation of "Cross-Domain Adaptive Clustering for Semi-Supervised Domain Adaptation" accepted by CVPR2021.
More details of this work can be found in our paper: [[Arxiv]](https://arxiv.org/abs/2104.09415) or [[OpenAccess]](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Cross-Domain_Adaptive_Clustering_for_Semi-Supervised_Domain_Adaptation_CVPR_2021_paper.html).

Our code is based on [SSDA_MME](https://github.com/VisionLearningGroup/SSDA_MME) implementation.

## Install

`pip install -r requirements.txt`

The code is written in Python 3.8.5, but should work for other versions with some modifications.


## Data preparation

Refer to [SSDA_MME](https://github.com/VisionLearningGroup/SSDA_MME) and our paper.

## Training
(1) To run training on DomainNet in the 3-shot scenario using alexnet,

`python main.py --dataset multi --source real --target sketch --net alexnet --num 3 --lr_f 1.0 --multi 0.1 --save_check`

(2) To run training on Office-Home in the 3-shot scenario using alexnet,

`python main.py --dataset office_home --source Real --target  Art --net alexnet --num 3 --lr_f 1.0 --multi 0.1  --steps 20000 --save_check`


(3) To run training on Office31 in the 3-shot scenario using alexnet,

`python main.py --dataset office --source webcam --target amazon --net alexnet --num 3 --lr_f 1.0 --multi 0.1 --steps 5000 --save_check`


### Citation
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

### Contact
Please feel free to contact the first author, namely [Li Jichang](https://lijichang.github.io/), with an Email address li.jichang@foxmail.com, if you have any questions.
