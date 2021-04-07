# PyTorch Semantic Segmentation

> A simple PyTorch implementation of semantic segmentation models.

This repository aims to implement semantic segmentation models with PyTorch.

## Build Status

![Python version](https://img.shields.io/badge/Python-3.8-orange) ![PyTorch version](https://img.shields.io/badge/PyTorch-1.8-brightgreen) ![GitHub release (latest by date)](https://img.shields.io/github/v/release/synml/pytorch-semantic-segmentation) ![GitHub last commit](https://img.shields.io/github/last-commit/synml/pytorch-semantic-segmentation) [![GitHub license](https://img.shields.io/github/license/synml/pytorch-semantic-segmentation)](https://github.com/synml/pytorch-semantic-segmentation/blob/main/LICENSE)

## Table of Contents

- [Requirements](#Requirements)
- [Models](#Models)
- [Datasets](#Datasets)
- [How to Use](#How to Use)
- [Module Description](#Module Description)
- [Credits](#Credits)
- [Contribution](#Contribution)
- [License](#License)

## Requirements

- Hardware (Developer environment)
  - CPU: Intel Core i7 9700
  - RAM: 32GiB
  - GPU: Nvidia Geforce RTX 3090
  - Storage: Samsung SSD 970 Pro 512GB
- Software
  - OS: Ubuntu (Primary), Windows (Secondary)
  - Miniconda (Python 3.8)
  - PyTorch 1.8.1 (CUDA 11.1)
- Dependent Packages
  - Matplotlib
  - PyYAML
  - Scikit-learn
  - Tensorboard
  - Tqdm
- Useful Packages
  - [pytorch-summary](https://github.com/sksq96/pytorch-summary)
  - [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch)
  - [Netron](https://github.com/lutzroeder/Netron) : Visualizer for neural network models. (web version: [Netron](https://lutzroeder.github.io/netron/))

## **Models**

This repository supports these semantic segmentation models as follows:

- (U-Net) Convolutional Networks for Biomedical Image Segmentation [Paper]
- (AR U-Net) Atrous Residual U-Net for Semantic Segmentation in Urban Street Scenes [Paper]
- (DeepLab v3) Rethinking Atrous Convolution for Semantic Image Segmentation [Paper]
- (DeepLab v3+) Encoder-Decoder with Atrous Separable Convolution for Semantic Segmentation [Paper]

## Dataset

This repository supports these datasets as follows:

- [Cityscapes](https://www.cityscapes-dataset.com/)
  
  1. Download dataset files. (*leftImg8bit_trainvaltest.zip* and *gtFine_trainvaltest.zip*).
  
  2. Extract downloaded files. The structure of dataset is as follows:
  
     ```
     |-- data
     |  |-- cityscapes
     |  |  |-- gtFine
     |  |  |  |-- test
     |  |  |  |-- train
     |  |  |  |-- val
     |  |  |-- leftImg8bit
     |  |  |  |-- test
     |  |  |  |-- train
     |  |  |  |-- val
     |  |  |-- license.txt
     |  |  |-- README
     ```
  
  3. Download [**cityscapesScripts**](https://github.com/mcordts/cityscapesScripts) for inspection, preparation, and evaluation. (Or clone [this repo](https://github.com/synml/cityscapesScripts))
  
  4. Edit the script `labels.py` to specify the label number.
  
  5. Edit the script `createTrainIdLabelImgs.py` to set cityscapes path.
  
  6. Run the script `createTrainIdLabelImgs.py` to create annotations based on the training labels.
- [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)

## How to Use

1. Clone this repository.
   
   - ```bash
     git clone https://github.com/synml/pytorch-semantic-segmentation
     ```
   
2. Create and activate a new virtual environment with Miniconda.

   - ```bash
      conda create -n [env_name, ex: torch] python=3.8
      ```
   - ```bash
      conda activate [env_name, ex: torch]
      ```

3. Install PyTorch.

   - https://pytorch.org/get-started/locally/

4. Install the dependent packages mentioned above.

   - ```bash
     conda install matplotlib pyyaml scikit-learn tensorboard tqdm
     ```

5. Prepare datasets.

6. Customize the configuration file. (**config.yaml**)

## Module Description

backup.py

clean.py

demo.py

eval.py

exec_tensorboard.py

featurte_visualizer.py

train.py

train_interupter.ini

## Credits

- https://github.com/meetshah1995/pytorch-semseg
- https://github.com/warmspringwinds/pytorch-segmentation-detection
- https://github.com/xiaoyufenfei/Efficient-Segmentation-Networks
- https://github.com/milesial/Pytorch-UNet

## Contribution method

1. Fork this repository.
2. Create a new branch or use the master branch.
3. Commit modifications.
4. Push on the selected branch.
5. Please send a pull request.

## License

You can find more information in `LICENSE`.