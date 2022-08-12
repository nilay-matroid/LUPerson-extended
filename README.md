# Matroid-LUPerson-extended

# Introduction
This is a modified implementation of Unsupervised Pre-training for Person Re-identification (LUPerson) which is used for Appearance based Similarity search in Matroid. \
The model is evaluated on three datasets, namely - Market1501, LaST and SYSU-30k. This documentation covers how to evaluate the model on all of these datasets as well as train the model on LaST and Market1501 dataset. It also covers steps to train and evaluate on a new dataset in the future.


# Environment
Set up the conda environment
```bash
conda env create -f environment.yml
conda activate fastreid
```

# Config file
The main parameters and arguments for training/evaluation are specified in config.yaml file such as 
- Base model architecture
- Backbones
- input/output sizes
- Batch sizes
- Reranking
- Caching
- Parallelized evaluation

Checkout fast-reid/fastreid/config/defaults.py for comprehensive list of parameters set by config files.

Some important ones are:
- Model
```bash
MODEL:
  BACKBONE:
    WITH_IBN: False
    EXTRA_BN: True
  PIXEL_MEAN: [89.896, 79.200, 80.073]
  PIXEL_STD: [63.872, 64.305, 63.839]
```
- Dataset
```bash
DATASETS:
  NAMES: ("LaST",)
  TESTS: ("LaST",)
  KWARGS: 'normalize:False+train_mode:False+verbose:True'
  ROOT: "./datasets/last"
```

- Test options
```bash
TEST:
  EVAL_PERIOD: 60
  IMS_PER_BATCH: 128
```
If you want to enable re-ranking. Warning: slower and takes lot of CPU memory but more accurate results.
```bash
TEST:
  RERANK:
    ENABLED: True
    K1: 20
    K2: 6
    LAMBDA: 0.3
```

If you want to cache the generated features
```bash
TEST:
  CACHE:
    ENABLED: True
    CACHE_DIR: "logs/lup_moco/test/last/cache"
```

If you want to run eval on cached features with multiple worker threads on different cpus.
```bash
TEST:
  CACHE:
    REUSE_FEAT: True
    PARALLEL:
      ENABLED: True
      NUM_WORKERS: 12
```

# Evaluation
Set variables
```bash
DATASET=market
PATH_TO_CHECKPOINT_FILE='<CHECKPOINT_PATH>'
```

Run eval script 
```bash
python tools/train_net.py --eval-only --config-file <CONFIG_FILE_PATH> DATASETS.ROOT "datasets" DATASETS.KWARGS "data_name:${DATASET}" MODEL.WEIGHTS ${PATH_TO_CHECKPOINT_FILE} MODEL.DEVICE "cuda:0" OUTPUT_DIR "./logs/lup_moco/test/${DATASET}"
```

Specific settings for each dataset
- Market1501 \
  DATASET=market \
  --config-file "./configs/CMDM/mgn_R50_moco.yml"
  
- LaST \
  DATASET=last \
  --config-file "./configs/LaST/mgn_R50_moco_cache_test.yml" to generate cached features \
  --config-file "./configs/LaST/mgn_R50_moco_cache_reuse_parallelized_test.yml" to evaluate cached features

- SYSU-30k \
  DATASET=last \
  --config-file "./configs/SYSU-30k/mgn_R50_moco_cache_test.yml"


# Training

Script to train on LaST dataset.

```bash
python tools/train_net.py --num-gpus 1 --config-file ./configs/LaST/mgn_R50_moco_train.yml SOLVER.CHECKPOINT_PERIOD 5
```

# LUPerson (Original Paper documentation)
Unsupervised Pre-training for Person Re-identification (LUPerson).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-pre-training-for-person-re/person-re-identification-on-msmt17)](https://paperswithcode.com/sota/person-re-identification-on-msmt17?p=unsupervised-pre-training-for-person-re)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-pre-training-for-person-re/person-re-identification-on-dukemtmc-reid)](https://paperswithcode.com/sota/person-re-identification-on-dukemtmc-reid?p=unsupervised-pre-training-for-person-re)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-pre-training-for-person-re/person-re-identification-on-market-1501)](https://paperswithcode.com/sota/person-re-identification-on-market-1501?p=unsupervised-pre-training-for-person-re)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-pre-training-for-person-re/person-re-identification-on-cuhk03-labeled)](https://paperswithcode.com/sota/person-re-identification-on-cuhk03-labeled?p=unsupervised-pre-training-for-person-re)

The repository is for our CVPR2021 paper [Unsupervised Pre-training for Person Re-identification](https://arxiv.org/abs/2012.03753).

## LUPerson Dataset
LUPerson is currently the largest unlabeled dataset for Person Re-identification, which is used for Unsupervised Pre-training. LUPerson consists of 4M images of over 200K identities and covers a much diverse range of capturing environments. 

**Details can be found at ./LUP**.

## Pre-trained Models
| Model | path |
| :------: | :------: |
| ResNet50 | [R50](https://drive.google.com/file/d/1pFyAdt9BOZCtzaLiE-W3CsX_kgWABKK6/view?usp=sharing) |
| ResNet101 | [R101](https://drive.google.com/file/d/1Ckn0iVtx-IhGQackRECoMR7IVVr4FC5h/view?usp=sharing) |
| ResNet152 | [R152](https://drive.google.com/file/d/1nGGatER6--ZTHdcTryhWEqKRKYU-Mrl_/view?usp=sharing) |

## Finetuned Results
For MGN with ResNet50:

|Dataset | mAP | cmc1 | path |
|:------:|:---:|:----:|:----:|
| MSMT17 | 66.06/79.93 | 85.08/87.63 | [MSMT](https://drive.google.com/file/d/1bV27gwAsX8L3a3yhLoxAJueqrGmQTodV/view?usp=sharing) |
| DukeMTMC | 82.27/91.70 | 90.35/92.82 | [Duke](https://drive.google.com/file/d/1leUezGnwFu8LKG2N8Ifd2Ii9utlJU5g4/view?usp=sharing) |
| Market1501 | 91.12/96.16 | 96.26/97.12 | [Market](https://drive.google.com/file/d/1AlXgY5bI0Lj7HClfNsl3RR8uPi2nq6Zn/view?usp=sharing) |
| CUHK03-L | 74.54/85.84 | 74.64/82.86 | [CUHK03](https://drive.google.com/file/d/1BQ-zeEgZPud77OtliM9md8Z2lTz11HNh/view?usp=sharing)|

These numbers are a little different from those reported in our paper, and most are slightly better.

For MGN with ResNet101:
|Dataset | mAP | cmc1 | path |
|:------:|:---:|:----:|:----:|
| MSMT17 | 68.41/81.12 | 86.28/88.27 | - |
| DukeMTMC | 84.15/92.77 | 91.88/93.99 | - |
| Market1501 | 91.86/96.21 | 96.56/97.03 | - |
| CUHK03-L | 75.98/86.73 | 75.86/84.07 | - |

**The numbers are in the format of `without RR`/`with RR`**.


## Citation
If you find this code useful for your research, please cite our paper.
```
@article{fu2020unsupervised,
  title={Unsupervised Pre-training for Person Re-identification},
  author={Fu, Dengpan and Chen, Dongdong and Bao, Jianmin and Yang, Hao and Yuan, Lu and Zhang, Lei and Li, Houqiang and Chen, Dong},
  journal={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2021}
}
```

## News
We extend our `LUPerson` to `LUPerson-NL` with `Noisy Labels` which are generated from tracking algorithm, Please check for our CVPR22 paper [Large-Scale Pre-training for Person Re-identification with Noisy Labels](https://arxiv.org/abs/2203.16533). And LUPerson-NL dataset is available at https://github.com/DengpanFu/LUPerson-NL
