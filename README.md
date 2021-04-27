# FourierHeatmap (latest release: v0.2.0)

[![CI](https://github.com/gatheluck/FourierHeatmap/workflows/CI/badge.svg)](https://github.com/gatheluck/FourierHeatmap/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/gatheluck/FourierHeatmap/branch/master/graph/badge.svg?token=17KIZNS046)](https://codecov.io/gh/gatheluck/FourierHeatmap)
[![MIT License](https://img.shields.io/github/license/gatheluck/FourierHeatmap)](LICENSE)

This is an unofficial pytorch implementation of Fourier Heat Map which is proposed in the paper, [A Fourier Perspective on Model Robustness in Computer Vision [Yin+, NeurIPS2019]](https://arxiv.org/abs/1906.08988). 

Fourier Heat Map allows to investigate the sensitivity of CNNs to high and low frequency corruptions via a perturbation analysis in the Fourier domain.

<img src="samples/FourierHeatmap-Teaser.png" height="300px">

## News
- We release v0.2.0. API is renewed and some useful libraries (e.g. [hydra](https://hydra.cc/docs/intro/)) are added.

- Previous version is still available as [v0.1.0](https://github.com/gatheluck/FourierHeatmap/tree/v0.1.0).

- [Docker is supported](#Evaluating_Fourier_Heat_Map_through_Docker). Now, you can evaluate Fourier Heat Map on the Docker container.

## Requirements
This library requires following as a pre-requisite.
- python 3.9+
- poetry

Note that I run the code with Ubuntu 20, Pytorch 1.8.1, CUDA 11.0.

## Installation
This repo uses [poetry](https://python-poetry.org/) as a package manager. 
The following code will install all necessary libraries under `.venv/`.

```
$ git clone git@github.com:gatheluck/FourierHeatmap.git
$ cd FourierHeatmap
$ pip install poetry  # If you haven't installed poetry yet.
$ poetry install
```

## Setup

### Dataset

This codes expect datasets exist under `data/`. For example, if you want to evaluate Fourier Heat Map for ImageNet, please set up like follows:

```
FourierHeatmap
├── data
│	└── imagenet
│		├── train/
│		└── val/
```

## Usage

### Visualizing Fourier basis

The script `fhmap/fourier/basis.py` generates Fourier base functions. For example:

```
$ poetry run python fhmap/fourier/basis.py
```

will generate 31x31 2D Fourier basis and save as an image under `outputs/basis.png`. The generated image should be like follows. 

<img src="samples/basis_31x31.png" height="300px">

### Evaluating Fourier Heat Map

The script `fhmap/apps/eval_fhmap.py`
eveluate Fourier Heat Map for a model. For example:

```
$ poetry run python fhmap/apps/eval_fhmap.py dataset=cifar10 arch=resnet56 weightpath=[PYTORCH_MODEL_WEIGHT_PATH] eps=4.0
```

will generate 31x31 Fourier Heat Map for ResNet56 on CIFAR-10 dataset and save as an image under `outputs/eval_fhmap/`. The generated image should be like follows. 

<img src="samples/cifar10_resnet56_natural.png" height="300px">


Note that the L2 norm size (=eps) of Fourier basis use in original paper is following:
| dataset | eps
---- | ----
| CIFAR-10 | 4.0
| ImageNet | 15.7

## Evaluating custom dataset and model



### Evaluating your custom dataset 

If you want to evaluate Fourier Heat Map on your custom dataset, please refer follwing instraction.

- Implement `YourCustomDatasetStats` class: 
	- This class holds basic dataset information.
	- `YourCustomDatasetStats` class should inherit from original `DatasetStats` class in `factory/dataset` module and also shoud be placed in `factory/dataset` module. 
	- For details, please refer to the `Cifar10Stats` class in `factory/dataset` module.

- Implement `YourCustomDataModule` class:
	- This class is responsible for preprocess, transform (includes adding Fourier Noise to image) and create test dataset.
	- `YourCustomDataModule` class should inherit from `BaseDataModule` class in `factory/dataset` module and also shoud be placed in `factory/dataset` module. 
	- For details, please refer to the `Cifar10DataModule` class in `factory/dataset` module.

- Implement `YourCustomDatasetConfig` class:
	- This class is needed for applying [hydra](https://hydra.cc/)'s [dynamic object instantiation](https://hydra.cc/docs/patterns/instantiate_objects/overview) to dataset class.
	- `YourCustomDatasetConfig` class should inherit from `DatasetConfig` class in `schema/dataset` module and also shoud be placed in `schema/dataset` module. Please add `YourCustomDatasetConfig` to `schema/__init__`.
	- For details, please refer to the `Cifar10Config` class in `schema/dataset` module.

- Add option for your custom dataset:
	- Lastly, please add the config of your custom dataset to `ConfigStore` class by adding a follwing line to `apps/eval_fhmap`.

	```
	cs.store(group="dataset", name="yourcustomdataset", node=schema.YourCustomDatasetConfig)
	```

Now, you will be able to call your custom dataset like following.

```
$ poetry run python fhmap/apps/eval_fhmap.py dataset=yourcustomdataset arch=resnet50 weightpath=[PYTORCH_MODEL_WEIGHT_PATH] eps=4.0
```

### Evaluating your custom architecture (model)

If you want to evaluate Fourier Heat Map on your custom architecture (model), please refer follwing instraction.

- Implement `YourCustomArch` class:
	- Please implement class or function which return your custom architecture. The custom architecture have to subclass of `torch.nn.module`.
	- For details, please refer to the `factory/archs/resnet` module.

- Implement `YourCustomArchConfig` class:
	- This class is needed for applying [hydra](https://hydra.cc/)'s [dynamic object instantiation](https://hydra.cc/docs/patterns/instantiate_objects/overview) to architecture class.
	- `YourCustomArchConfig` class should inherit from `ArchConfig` class in `schema/arch` module and also shoud be placed in `schema/arch` module. Please add `YourCustomArchConfig` to `schema/__init__`.
	- For details, please refer to the `Resnet56Config` class in `schema/arch` module.
	- If you want to use architectures which is provided by other libs like [pytorch](https://github.com/pytorch/pytorch) or [timm](https://github.com/rwightman/pytorch-image-models), please refere to the `Resnet50Config` class in `schema/arch` module.

- Add option for your custom architecture:
	- Lastly, please add the config of your custom architecture to `ConfigStore` class by adding a follwing line to `apps/eval_fhmap`.

	```
	cs.store(group="arch", name="yourcustomarch", node=schema.YourCustomArchConfig)
	```

Now, you will be able to call your custom arch like following.

```
$ poetry run python fhmap/apps/eval_fhmap.py dataset=cifar10 arch=yourcustomarch weightpath=[PYTORCH_MODEL_WEIGHT_PATH] eps=4.0
```

## Evaluating Fourier Heat Map through Docker
In order to use FourierHeatmap throgh docker, please install Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html) beforehand. For detail, please refere [official installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installation-guide).

If `nvidia-smi` is able to run through docker like following, it is successfully installed.

```
$ sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

Tue Apr 27 06:46:09 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.102.04   Driver Version: 450.102.04   CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce GTX 1080    Off  | 00000000:01:00.0  On |                  N/A |
| N/A   56C    P0    42W /  N/A |   1809MiB /  8114MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+

```
We use environmental variables to specify the arguments.
The variables that can be specified and their meanings are as follows:

| name | optional | default | description
---- | ---- | ---- | ----
| HOST_DATADIR    | False | | Path to the directory where the dataset is located in the host.
| HOST_OUTPUTSDIR | False | | Path to the directory where the output will be located in the host.
| HOST_WEIGHTDIR  | False | | Path to the directory where the pretrained wight is located in the host.
| WEIGHTFILE      | False | | File name of the pretrained wight.
| ARCH       | True | resnet56 | Name of the architecture.
| BATCH_SIZE | True | 512      | Size of batch.
| DATASET    | True | cifar10  | Name of dataset.
| EPS        | True | 4.0      | L2 norm size of Fourier basis.
| IGNORE_EDGE_SIZE  | True | 0 | Size of the edge to ignore.
| NUM_SAMPLES       | True | -1| Number of samples used from dataset. If -1, use all samples.
| NVIDIA_VISIBLE_DEVICES   | True | 0 | Device number (or list of number) visible from CUDA.

For example:

```
$ export HOST_DATADIR=[DATASET_DIRECTORY_PATH]
$ export HOST_OUTPUTSDIR=[OUTPUTS_DIRECTORY_PATH]
$ export HOST_WEIGHTDIR=[WEIGHT_DIRECTORY_PATH]
$ export WEIGHTFILE=[PYTORCH_MODEL_FILE]
$ cd provision/docker
$ sudo -E docker-compose up  # -E option is needed to inherit environment variables.
```

will generate 31x31 Fourier Heat Map for ResNet56 on CIFAR-10 dataset and save as an image under `OUTPUTS_DIRECTORY_PATH/eval_fhmap/`.

## References

- [Dong Yin, Raphael Gontijo Lopes, Jonathon Shlens, Ekin D. Cubuk, Justin Gilmer. "A Fourier Perspective on Model Robustness in Computer Vision.", in NeurIPS, 2019.](https://arxiv.org/abs/1906.08988)
- [Justin Gilmer and Dan Hendrycks. "Adversarial Example Researchers Need to Expand What is Meant by ‘Robustness’.", in Distill, 2019.](https://distill.pub/2019/advex-bugs-discussion/response-1/)