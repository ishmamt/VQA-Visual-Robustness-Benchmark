# Visual Robustness Benchmark for Visual Question Answering (VQA)

[![arXiv](https://img.shields.io/badge/arXiv-2407.03386-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2407.03386)

[Md Farhan Ishmam*](https://farhanishmam.github.io/), [Ishmam Tashdeed*](https://cse.iutoic-dhaka.edu/profile/ishmamtashdeed/), [Talukder Asir Saadat*](https://www.bubt.edu.bd/department/member_details/806), [Md. Hamjajul Ashmafee](https://cse.iutoic-dhaka.edu/profile/ashmafee/), [Md. Azam Hossain](https://cse.iutoic-dhaka.edu/profile/azam/), and [Abu Raihan Mostofa Kamal](https://cse.iutoic-dhaka.edu/profile/raihan-kamal/) (*Equal Contribution)

## Overview of the Visual Robustness Benchmark

![image](./assets/overview.png)

## Quick Start

[<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1gTsUG5BNp3MPyQQS8L6qpBqpZD45E3Vp?usp=sharing)

**Note:** Make sure that you already have the `Image directory`, `question JSON`, and `annotation JSON` in your drive.

## Installation

This repository is tested on `Python 3.8`. Create a virtual environment to install all the dependencies.

### Step 1:
To install `torch` visit the <a href="https://pytorch.org/get-started/locally/">PyTorch</a> website and follow the instructions.

### Step 2:
Install `requirements.txt` file running the command:
```
pip install -r requirements.txt
```

### Step 3:
Install `MagickWand` library. If you have a **Linux** system use:
```
sudo apt-get install libmagickwand-dev
```
If you have a **Windows** system, follow steps given in the <a href='https://docs.wand-py.org/en/latest/guide/install.html#install-imagemagick-on-windows:~:text=/opt/local-,Install%20ImageMagick%20on%20Windows,-%C2%B6'>website</a>.

## Dataset

This repository uses a random split of the VQAv2 dataset found on the <a href="https://visualqa.org/download.html">Visual Question Answering</a> website.

## Experiment

To run the experiment, execute the following command:
```
python main.py
```

## Data Analysis

Data analysis for this work is done in Python via the given notebooks. All the plots included in the paper can be reproduced via the notebooks.

## Acknowledgement

We thank the Network and Data Analysis Group (NDAG), IUT-CSE for providing the computational resources to conduct the experiments.

## Citation
```
@article{ishmam2024visual,
  title={Visual Robustness Benchmark for Visual Question Answering (VQA)},
  author={Ishmam, Md Farhan and Tashdeed, Ishmam and Saadat, Talukder Asir and Ashmafee, Md Hamjajul and Kamal, Dr Abu Raihan Mostofa and Hossain, Dr Md Azam},
  journal={arXiv preprint arXiv:2407.03386},
  year={2024}
}
```
