#!/bin/sh

#
# Please note the license conditions of this software / dataset!
#

# left images
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
# right images
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_3.zip
# velodyne
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip
# calibration
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip
# labels
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
# unzip all 
for z in *.zip; do unzip $z; done