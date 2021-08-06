# import packages
import os
import sys
import torch
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from dataset import Dataset
from torchvision import transforms, utils

# add path to sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.dirname(os.path.abspath('src/utils'))
sys.path.append(BASE_DIR)
sys.path.append(UTILS_DIR)

from utils import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class KITTIObjectDataset(Dataset):

    def __init__(self, dataroot, n_xgrids, n_ygrids, mean_lwh, mode='train', 
                       xlim=(0.0, 70.0), ylim=(-40.0, 40.0), zlim=(-2.5, 1.0),
                       vol_size=(512,512,32), img_size=(256,256)):
        """
        Args:
            dataroot (string): Root Directory of KITTI object dataset.
        """
        # paths
        self.dataroot = dataroot
        if mode == 'train':
            self.train_dir = os.path.join(dataroot, 'training')
        elif mode == 'test':
            self.train_dir = os.path.join(dataroot, 'testing')
        self.left_image_dir = os.path.join(self.train_dir, 'image_2')
        self.right_image_dir = os.path.join(self.train_dir, 'image_3')
        self.label_dir = os.path.join(self.train_dir, 'label_2')
        self.velodyne_dir = os.path.join(self.train_dir, 'velodyne')

        # axes limits
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim

        # volume size
        self.vol_size = vol_size

        # image size
        self.img_size = img_size

        # filenames
        self.left_image_filenames = []
        self.right_image_filenames = []
        self.label_filenames = []
        self.velodyne_filenames = []
        self.label_dict_list = []

        # labels
        self.n_xgrids = n_xgrids
        self.n_ygrids = n_ygrids

        # mean lwh
        self.mean_lwh = mean_lwh

        # camera intrinsic matrix
        self.K = np.array([[7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 4.485728000000e+01],
                           [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 2.163791000000e-01],
                           [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 2.745884000000e-03]])

        # lidar-to-camera extrinsic matrix
        self.T_lidar2cam = [[ 0.0002, -0.9999, -0.0106,  0.0594],
                            [ 0.0104,  0.0106, -0.9999, -0.0751],
                            [ 0.9999,  0.0001,  0.0105, -0.2721],
                            [ 0.,      0.,      0.,      1.    ]]

        # configs needed for stereo
        self.fx = self.K[0,0]
        self.baseline = 0.54 # meters
        self.max_depth = 200.0

        # image resolution
        self.img_resolution = (1242, 375)

        # list files
        img_fnames_list = sorted(os.listdir(self.left_image_dir))

        # go through each files
        for img_fname in img_fnames_list:
            fname, file_ext = os.path.splitext(img_fname)
            
            # construct filenames
            left_image_full_path = os.path.join(self.left_image_dir, img_fname)
            right_image_full_path = os.path.join(self.right_image_dir, fname + '.png')
            label_full_path = os.path.join(self.label_dir, fname + '.txt')
            velodyne_full_path = os.path.join(self.velodyne_dir, fname + '.bin')

            # check if files exist
            if os.path.exists(left_image_full_path) and \
               os.path.exists(right_image_full_path) and \
               os.path.exists(label_full_path) and \
               os.path.exists(velodyne_full_path):

                # append to list
                self.left_image_filenames.append(left_image_full_path)
                self.right_image_filenames.append(right_image_full_path)
                self.label_filenames.append(label_full_path)
                self.velodyne_filenames.append(velodyne_full_path)

                # get labels
                labels_str = None
                label_dict_list_this_frame = []
                with open(label_full_path, 'r') as file:
                    labels_str = file.readlines()

                    # go through each object
                    for i, label in enumerate(labels_str):
                        label = label.replace('\n', '').split()

                        # parse fields
                        class_obj = label[0]
                        truncated = float(label[1])
                        occlusion = int(label[2]) # 0 = fully visible, 1 = partly occluded
                                                  # 2 = largely occluded, 3 = unknown

                        h = float(label[8])
                        w = float(label[9])
                        l = float(label[10])
                        # these are in camera coordinate system
                        x_cam = float(label[11])
                        y_cam = float(label[12]) - (h/2.0)
                        z_cam = float(label[13])
                        orientation = np.pi-float(label[14])

                        # converting to lidar coordinate system
                        x = z_cam
                        y = -x_cam
                        z = -y_cam

                        # check if the object is within range
                        if ((x > xlim[0]) and (x < xlim[1]) and \
                            (y > ylim[0]) and (y < ylim[1]) and \
                            (z > zlim[0]) and (z < zlim[1])):

                            if class_obj in label_map.keys():
                                label_dict = {}
                                label_dict['conf'] = 1.0
                                label_dict['x'] = x
                                label_dict['y'] = y
                                label_dict['z'] = z
                                label_dict['l'] = l
                                label_dict['w'] = w
                                label_dict['h'] = h
                                label_dict['yaw'] = orientation
                                label_dict['class'] = class_obj
                                label_dict_list_this_frame.append(label_dict)

                # append labels 
                self.label_dict_list.append(label_dict_list_this_frame)

        # print info
        print('Number of data samples found in the directory: {}'.format(len(self.left_image_filenames)))

    # method to get length of data
    def __len__(self):
        return len(self.left_image_filenames)

    # method to obtain camera intrinsics
    def camera_intrinsics(self):
        return self.K

    # method to obtain original image resolution [x, y]
    def image_resolution(self):
        return self.img_resolution

    # method to obtain lidar to camera extrinsic matrix
    def lidar2cam(self):
        return self.T_lidar2cam

    # method to obtain mean [length, width, height]
    def get_mean_lwh(self):
        return self.mean_lwh

# main function
if __name__ == '__main__':
    from utils import *

    mean_lwh = {'Car': [3.8840, 1.6286, 1.5261], 'Pedestrian': [1.7635, 0.5968, 1.7372], 'Cyclist': [0.8423, 0.6602, 1.7607]}
    dataset = KITTIObjectDataset('/floppy/datasets/KITTI/object', \
                                 n_xgrids=16, n_ygrids=16, xlim=(0,70), vol_size=(256,256,16), mean_lwh=mean_lwh)

    # visualization window
    cv2.namedWindow('VR3Dense', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('VR3Dense', 900, 1440)
    # show 100 samples
    for i in range(100): 
        sample = dataset[i]
        # build label vector and then recover poses and classes
        label_vec = build_label_vector(sample['label_dict'], dataset.n_xgrids, dataset.n_ygrids, dataset.mean_lwh, xlim=dataset.xlim, ylim=dataset.ylim, zlim=dataset.zlim)
        label_dict = decompose_label_vector(label_vec, dataset.n_xgrids, dataset.n_ygrids, dataset.mean_lwh, xlim=dataset.xlim, ylim=dataset.ylim, zlim=dataset.zlim)
        # recover point-cloud from voxelized volume
        pc_voxelized = volume_to_point_cloud(sample['cloud_voxelized'], vol_size=(256,256,16), xlim=dataset.xlim, ylim=dataset.ylim, zlim=dataset.zlim)

        # draw point-cloud
        canvasSize = 1200
        pc_bbox_img = draw_point_cloud_w_bbox(pc_voxelized, label_dict, canvasSize=canvasSize, xlim=dataset.xlim, ylim=dataset.ylim, zlim=dataset.zlim)

        # get labels in camera coordinate system
        label_cam = label_lidar2cam(label_dict, dataset.T_lidar2cam)
        # draw bounding box on image
        img_rgb = np.transpose(sample['left_image'], (1,2,0))
        img_rgb = draw_bbox_img(img_rgb, label_cam, dataset.K)

        # resize image
        scale_factor = canvasSize / img_rgb.shape[1] 
        width = int(img_rgb.shape[1] * scale_factor)
        height = int(img_rgb.shape[0] * scale_factor)
        img_rgb = cv2.resize(img_rgb, (width, height), interpolation = cv2.INTER_AREA) 
        img_rgb = np.array(img_rgb, dtype=np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        pc_bbox_img_bgr = cv2.cvtColor(pc_bbox_img, cv2.COLOR_RGB2BGR)
        pc_bbox_img_bgr = np.array(pc_bbox_img_bgr*255.0, dtype=np.uint8)

        # points on image
        projected_pts = np.squeeze(sample['lidar_cam_projection'], 0)
        projected_pts = colorize_depth_map(projected_pts, mask_zeros=True)
        projected_pts = cv2.resize(projected_pts, (width, height), interpolation = cv2.INTER_NEAREST) 
        projected_pts = cv2.cvtColor(projected_pts, cv2.COLOR_RGB2BGR)
        projected_pts = np.array(projected_pts, dtype=np.uint8)
        projected_pts = draw_bbox_img(projected_pts, label_cam, dataset.K)

        # concat image with point-cloud 
        img_viz = cv2.vconcat([img_bgr, projected_pts, pc_bbox_img_bgr])

        cv2.imshow('VR3Dense', img_viz)
        cv2.waitKey(0)