# Classification with LIO

This page provides basic tutorials about the usage of Look-into-Object for image classfication.

This code is tested with PyTorch 1.4.0 and torchvision 0.4.1.

## Setup
### Install dependencies
```bash
python -m pip install -r requirements.txt
```

### Prepare dataset
You can follow the `Datasets Prepare` Section in [DCL](https://github.com/JDAI-CV/DCL#datasets-prepare).

Note: The `label_num` in annotations starts from 1 rather than 0.

## Train a model
Run `train_index.py` to train CUB/STCAR/AIR.

Train with last stage and 3 positive images on CUB (LIO/ResNet-50 7x7):
```bash
python train_index.py --data CUB --stage 3 --num_positive 3
```

## Help

Feel free to open an issue if you encounter troubles.

## Citation

If you use this codebase in your research, please cite our paper:

```bib
@InProceedings{Zhou_2020_CVPR,
author = {Zhou, Mohan and Bai, Yalong and Zhang, Wei and Zhao, Tiejun and Mei, Tao},
title = {Look-Into-Object: Self-Supervised Structure Modeling for Object Recognition},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

