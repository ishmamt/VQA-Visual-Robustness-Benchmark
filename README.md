# Visual Robustness
Framework for visual robustness testing of any VQA model. Work in progress.

<h2 id="getting-started">Getting Started</h2>

### Step 1:
**If you have CUDA support** then install `torch`, `torchvision` and `torchaudio` as:
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/[CUDA_VERSION]
```
**For details, go to: <a href="https://pytorch.org/get-started/locally/">PyTorch</a> website.**

**If you don't have CUDA support** then install `torch`, `torchvision` and `torchaudio` as:
```
pip install torch torchvision torchaudio
```

### Step 2:
Install `requirements.txt` file as:
```
pip install -r requirements.txt
```

### Step 3:
Install **MagickWand library**:

If you have `Linux` system use:
```
sudo apt-get install libmagickwand-dev
```

If you have a `Windows` system, follow steps given in the <a href='https://docs.wand-py.org/en/latest/guide/install.html#install-imagemagick-on-windows:~:text=/opt/local-,Install%20ImageMagick%20on%20Windows,-%C2%B6'>website</a>.


## Notebook
A helpful <a href='https://colab.research.google.com/drive/1gTsUG5BNp3MPyQQS8L6qpBqpZD45E3Vp?usp=sharing'>Notebook</a> is given to run basic commands. **You don't need to run the above mentioned <a href='#getting-started'> getting started </a> section as it is already done for you in the notebook.**

**Note:** Make sure that you already have the `Image directory`, `question JSON` and `annotation JSON` in your drive.

Follow the further instructions given in the notebook.
