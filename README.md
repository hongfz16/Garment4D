# Garment4D
[[PDF]](https://papers.nips.cc/paper/2021/file/eb160de1de89d9058fcb0b968dbbbd68-Paper.pdf)  |  [[OpenReview]](https://openreview.net/forum?id=aF60hOEwHP)  |  [[arXiv]](https://arxiv.org/abs/2112.04159)  |  [[Project Page]](https://hongfz16.github.io/projects/Garment4D.html)

## 1. Overview
This is the codebase for our NeurIPS 2021 paper *Garment4D: Garment Reconstruction from Point Cloud Sequences*.

![teaser](./assets/cover.jpg)

For further information, please contact [Fangzhou Hong](mailto:fangzhouhong820@gmail.com).

## 2. News
- **2021-12** Code release!
- **2021-09** Garment4D is accepted to NeurIPS 2021.

## 3. Getting Started

### 3.1 Setup the Environment
The code has been tested with `Python 3.7.9`, `PyTorch 1.5.0`, `CUDA 10.1`. Other required packages include:

- tqdm
- loguru
- yaml
- easydict
- numpy
- scipy
- [chamferdist](https://github.com/krrish94/chamferdist)
- [openmesh](https://pypi.org/project/openmesh/)
- [pytorch_scatter](https://github.com/rusty1s/pytorch_scatter)

Other than the above packages, please also compile the cuda kernels required by PointNet++ by executing `python setup.py install` under `modules/pointnet2/pointnet2`.

### 3.2 Download SMPL model files
Please register and download SMPL model files [here](https://smpl.is.tue.mpg.de/). Then please put the model files at `smplx/model`. The folder structure should be like

```
./
├── 
├── ...
└── smplx/
    ├── models/
        ├── smpl/
            ├── SMPL_MALE.pkl
            ├── SMPL_FEMALE.pkl
            └── SMPL_NEUTRAL.pkl
    ├── ...
```

### 3.3 Setup the Dataset
Please download the processed CLOTH3D dataset using the following links. Due the size of the whole dataset is big (~160 GB), we split the archieve into several 10 GB chunks. You could combine them by `cat CLOTH3D.z* > merged_CLOTH3D.zip` and unzip it and put it under `dataset`.

|   |   |   |   |
|---|---|---|---|
|[CLOTH3D.z01](https://1drv.ms/u/s!AjLpFg-f48lj0RbDpBWj2uGuMX2e?e=anROId)|[CLOTH3D.z02](https://1drv.ms/u/s!AjLpFg-f48lj0RXsELoQGQ6C-X3v?e=CVmeHQ)|[CLOTH3D.z03](https://1drv.ms/u/s!AjLpFg-f48lj0RSY0JPLCxrmy9mJ?e=PmFqY3)|[CLOTH3D.z04](https://1drv.ms/u/s!AjLpFg-f48lj0RfSoFICM2rUwKl2?e=UrNj4P)|
|[CLOTH3D.z05](https://1drv.ms/u/s!AjLpFg-f48lj0Rg7_EAigJgI7jDg?e=lsdPrA)|[CLOTH3D.z06](https://1drv.ms/u/s!AjLpFg-f48lj0RnmOpxZ9VA_Nv6J?e=osuugM)|[CLOTH3D.z07](https://1drv.ms/u/s!AjLpFg-f48lj0RxbJGY7Owx4nLcD?e=qZqZnH)|[CLOTH3D.z08](https://1drv.ms/u/s!AjLpFg-f48lj0RqeCYXk9T5lZqjT?e=4vASX9)|
|[CLOTH3D.z09](https://1drv.ms/u/s!AjLpFg-f48lj0RtehTkftJsd4Fq7?e=gSFMwP)|[CLOTH3D.z10](https://1drv.ms/u/s!AjLpFg-f48lj0R0nIXfFDtVi8XIl?e=DkU9HW)|[CLOTH3D.z11](https://1drv.ms/u/s!AjLpFg-f48lj0R5J-qiadrP9Egvd?e=r6dtLo)|[CLOTH3D.z12](https://1drv.ms/u/s!AjLpFg-f48lj0R8EehQlBGtUAHo9?e=MA0oB2)|
|[CLOTH3D.z13](https://1drv.ms/u/s!AjLpFg-f48lj0SDlMwT_C4OqBtc-?e=XbveLr)|[CLOTH3D.z14](https://1drv.ms/u/s!AjLpFg-f48lj0SL43ITohM6mgsiO?e=Vqu1ew)|[CLOTH3D.z15](https://1drv.ms/u/s!AjLpFg-f48lj0SHwwD3rTqCwKMH_?e=yU2Kse)|[CLOTH3D.zip](https://1drv.ms/u/s!AjLpFg-f48lj0SP3E2v4XejNz-_B?e=JrflX1)|

The folder structure should look like

```
./
├── 
├── ...
└── dataset/
    ├──CLOTH3D/
        ├── CLOTH3D/
        ├── CLOTH3D_template/
        ├── list/
        ├── Skirt_reg/
        ├── Trousers_reg/
        └── Tshirt_reg/
```

### 3.4 Download the Pre-train Models
Please download the pre-train models using the following links and put them under `pretrain`.

| Pre-train Model | Download Link | L2 Error on CLOTH3D (Posed) [mm] |
| --------------- | ------------- | --------------------------- |
| skirt.ckpt      | [link](https://1drv.ms/u/s!AjLpFg-f48lj0RFqLA2p0BTX-CAH?e=3YDmme) | 49.23 |
| trousers.ckpt   | [link](https://1drv.ms/u/s!AjLpFg-f48lj0ROr9We0BIw60MrX?e=3taehr) | 25.46 |
| tshirt.ckpt     | [link](https://1drv.ms/u/s!AjLpFg-f48lj0RKHye4YnXjHbyvr?e=XrMJoW) | 37.95 |

The folder structure should look like

```
./
├── 
├── ...
└── pretrain/
    ├── skirt.ckpt
    ├── trousers.ckpt
    └── tshirt.ckpt
```

### 3.5 Test with Pre-train models
Please checkout the `scripts/test` folder for the testing scripts. One should be able to run the corresponding testing scripts with the pre-train models and dataset setup correctly.

### 3.6 Train from Scratch
Please checkout the `scripts/train` folder for the training scripts. We currently support three types of garments i.e. skirts, Tshirts and trousers. Take skirts training as an example, please run the `train_skirt_canonical.sh` first for the canonical garment reconstruction and then run the `train_skirt_posed.sh` for the posed garment reconstruction.

## 4. License
Distributed under the MIT License. See `LICENSE` for more information.

## 5. Citation
If you find our work useful in your research, please consider citing the following papers:
```
@inproceedings{
    hong2021garmentd,
    title={Garment4D: Garment Reconstruction from Point Cloud Sequences},
    author={Fangzhou Hong and Liang Pan and Zhongang Cai and Ziwei Liu},
    booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
    year={2021},
    url={https://openreview.net/forum?id=aF60hOEwHP}
}
```

### Acknowledgments
In our implementation, we refer to the following open-source databases:
- [PointNet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch)
- [pygcn](https://github.com/tkipf/pygcn)
- [smplx](https://github.com/vchoutas/smplx)
