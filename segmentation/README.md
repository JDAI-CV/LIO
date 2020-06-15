# Segmentation with SCL

This page provides basic tutorials about the usage of MMDetection with SCL.

Current code is develop on [mmdetection@618dca08](https://github.com/open-mmlab/mmdetection/commit/618dca0895f5f4ede19f5feebb064648e128e12e).

## Setup

1. Install `mmdet` according to [this doc](https://github.com/open-mmlab/mmdetection/blob/master/docs/install.md), and run `python setup.py develop` in our repository.
2. Prepare datasets according to [this](https://github.com/open-mmlab/mmdetection/blob/master/docs/getting_started.md#prepare-datasets).
3. Create a directory `checkpoints/` in this folder and download pretrained models [R-50-C4 caffe 1x](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_r50_caffe_c4_1x-ea7d3428.pth) and [X-101-32x4d-FPN pytorch 1x](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_x101_32x4d_fpn_1x_20181218-44e635cc.pth) to `checkpoints/`.

## Train a model

### Train with multiple GPUs

```bash
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

More specifically, R-50-C4 can be trained with this command:

```shell
./tools/dist_train.sh configs/scl/mask_rcnn_r50_caffe_c4_1x.py 4 --validate
```

And X-101-32x4d-FPN can be trained with this command:

```shell
./tools/dist_train.sh configs/scl/mask_rcnn_x101_32x4d_fpn_1x.py 4 --validate
```

### Notice

- Current code isn't compatible with basic models, so training base models (e.g. R-50-C4 without SCL) with this code is not feasible.
- We trained with 4 GPUs and other configuration hasn't been tried.
- Evaluating is similar to the above operations.

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

