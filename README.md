# Visual Robustness
Framework for visual robustness testing of any VQA model. Work in progress.

## Getting Started
### Step 1:
**If you have CUDA support** then install `torch`, `torchvision` and `torchaudio` as:
```
# For Windows
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/[CUDA_VERSION]

# For Linux
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/[CUDA_VERSION]
```
For details, go to: <a href="https://pytorch.org/get-started/locally/">PyTorch</a> website.

**If you don't have CUDA support** then install `torch`, `torchvision` and `torchaudio` as:
```
pip install torch torchvision torchaudio
```
### Step 2:
Installing `transformers` from **HuggingFace**:
```
# For Windows
pip install transformers
# For Linux
pip install -q git+https://github.com/huggingface/transformers.git
```
### Step 3:
Install `requirements.txt` file as:
```
pip install -r requirements.txt
```
