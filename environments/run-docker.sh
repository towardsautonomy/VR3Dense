#!/bin/sh

## run the docker container with X11 socket forwarding
MNT_DIR=/vr3dense
# create a xauth file with access permission
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f /tmp/.docker.xauth-n nmerge -
# add non-network local connections to access control list
xhost +local:root
# run docker
sudo docker run -it --rm -e DISPLAY=unix$DISPLAY                     \
                         -v ${PWD}/../:${MNT_DIR}                    \
                         -v /media:/media                            \
                         -v /tmp/.X11-unix:/tmp/.X11-unix            \
                         -v /tmp/.docker.xauth:/tmp/.docker.xauth:rw \
                         -e XAUTHORITY=/tmp/.docker.xauth            \
                         --net=host                                  \
                         --gpus all towardsautonomy/environments:cuda-11.2-pytorch-1.8.1-tf-2.4.1-opencv
# remove non-network local connections from access control list
xhost -local:root
