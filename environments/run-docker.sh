#!/bin/sh

MNT_DIR=/vr3dense
sudo docker run -it --rm -v ${PWD}/../:${MNT_DIR} -v /media:/media --gpus all towardsautonomy/environments:cuda-11.2-pytorch-1.8.1-tf-2.4.1-opencv
