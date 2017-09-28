# Intro to Pytorch

## Setup

I use conda for managing Python versions and environments, it can be installed with from [here](https://conda.io/miniconda.html)

On Linux:
```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
source Miniconda3-latest-Linux-x86_64.sh
```

This code for this presentation is writen in Python 3.6. To create an environment with all the required dependancies and proper version of Python:

```
conda env create -f env.yml
```

Then to activate the environment:
```
source activate ml
```

To deactivate

```
source deactivate ml
```

If you have a computer with an Nvidia GPU you may want to setup GPU acceleration. A good tutorial can be found [here](https://medium.com/@vivek.yadav/deep-learning-setup-for-ubuntu-16-04-tensorflow-1-2-keras-opencv3-python3-cuda8-and-cudnn5-1-324438dd46f0)

## Data

In this talk I make use of [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist). To download the dataset (total size ~31MB):

```
# cd ~/data # where I keep my datasets
mkdir fashion-mnist

wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
```

## GPU vs CPU speeds

Times are for 5 training epochs of Fashion MNIST

|Batch Size|Hidden Size| Time taken CPU (s)| Time taken GPU (s)|
|---|--- |---    |---    |
|32 |256 |45.149 |16.819 |
|32 |512 |80.357 |17.196 |
|64 |512 |41.201 |9.639  |
|128|1024|49.248 |5.827  |
|256|2048|66.144 |4.066  |
|max|2048|28.104 |2.389  |

### Note: the version of `experiment.py` that I keep up to date can be found [here](https://github.com/erikreppel/PyTorch-tools)