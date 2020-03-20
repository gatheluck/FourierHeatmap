# FourierHeatmap

This is an unofficial pytorch implementation of Fourier Heat Map which is prosed in the paper, [A Fourier Perspective on Model Robustness in Computer Vision [Yin+, NeurIPS2019]](https://arxiv.org/abs/1906.08988). Fourier Heat Map allows to investigate the sensitivity of CNNs to high and low frequency corruptions via a perturbation analysis in the Fourier domain.

## Requirements

You will need the following to run the codes:
- Python 3.7+
- PyTorch 1.4+
- TorchVision

Note that I run the code with Ubuntu 18, Pytorch 1.4.0, CUDA 10.1