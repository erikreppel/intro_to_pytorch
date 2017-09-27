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

