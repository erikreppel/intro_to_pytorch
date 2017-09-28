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

If you're on Windows or MacOS it might be easiest to run in Docker, conda is not always portable based on package version numbers.

```
$ docker-compose build
$ docker-compose up
...
jupyter_1  |     Copy/paste this URL into your browser when you connect for the first time,
jupyter_1  |     to login with a token:
jupyter_1  |         http://localhost:8888/?token=00d56f0815766c1ee3dcbc0e0322894cfcc166556d6310ae
jupyter_1  | [I 20:36:45.808 NotebookApp] 302 GET /?token=00d56f0815766c1ee3dcbc0e0322894cfcc166556d6310ae (172.20.0.1) 0.68ms
jupyter_1  | [I 20:36:59.638 NotebookApp] Writing notebook-signing key to /root/.local/share/jupyter/notebook_secret
jupyter_1  | [W 20:36:59.639 NotebookApp] Notebook 2_creating_models.ipynb is not trusted
jupyter_1  | [I 20:37:00.107 NotebookApp] Kernel started: 22ffa729-38e5-48be-8c60-a713a361fe6a
jupyter_1  | [I 20:37:00.735 NotebookApp] Adapting to protocol v5.1 for kernel 22ffa729-38e5-48be-8c60-a713a361fe6a
```

Open the link with the token to auth Jupyter notebook.

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