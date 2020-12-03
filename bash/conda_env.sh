#!/usr/bin/env bash
# 0.1 Install CUDA and Cudnn: following https://medium.com/@exesse/cuda-10-1-installation-on-ubuntu-18-04-lts-d04f89287130
# 0.2 Install Anaconda: download anaconda *.sh --> bash *.sh

conda create -n nerabsa python=3.6

conda activate nerabsa

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

conda install -c anaconda scikit-learn

pip install --trusted-host repos.dev.chata.ai -i https://chata-pypi:juVWXXVXQV5ajuBA@repos.dev.chata.ai/simple/ Wombat==0.0.1
