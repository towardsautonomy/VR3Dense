# import packages
import os
import sys
import cv2
import numpy as np
from copy import deepcopy
import torch

# add path to sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.dirname(os.path.abspath('src/utils'))
sys.path.append(BASE_DIR)
sys.path.append(UTILS_DIR)

# import utilities
from utils import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# dataset class from which each dataset inherits
class Dataset(torch.utils.data.Dataset):

    def __init__(self):
        pass

    # method to get length of data
    def __len__(self):
        raise NotImplementedError

    # method to obtain camera intrinsics
    def camera_intrinsics(self):
        raise NotImplementedError

    # method to obtain original image resolution [x, y]
    def image_resolution(self):
        raise NotImplementedError

    # method to obtain mean [length, width, height]
    def get_mean_lwh(self):
        raise NotImplementedError

    # method to obtain lidar to camera extrinsic matrix
    def lidar2cam(self):
        raise NotImplementedError

    # method get object statistics
    def get_object_statistics(self):
        object_lwh = {}
        for object_str in label_map.keys():
            object_lwh[object_str] = []

        for label_dict in self.left_label_dict_list:
            for label_ in label_dict:
                object_lwh[label_['class']].append([label_['l'], label_['w'], label_['h']])

        return object_lwh

    # method print object statistics
    def print_object_statistics(self):
        object_lwh = self.get_object_statistics()
        print('=================================================================================')
        for object_str in object_lwh.keys():
            stats = np.array(object_lwh[object_str])
            print('{0:>12}: '.format(object_str), end='')
            print('Mean Length - {:.4f} | Mean Width - {:.4f} | Mean Height - {:.4f}'.format(
                                        np.mean(stats[:,0]), np.mean(stats[:,1]), np.mean(stats[:,2])))
        print('=================================================================================')

    # method to get each item
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        left_image_filename = self.left_image_filenames[idx]
        right_image_filename = self.right_image_filenames[idx]
        velodyne_filename = self.velodyne_filenames[idx]
        label_dict_list = self.label_dict_list[idx]
        label_vector = build_label_vector(label_dict_list, self.n_xgrids, self.n_ygrids, self.mean_lwh, \
                                            xlim=self.xlim, ylim=self.ylim, zlim=self.zlim)

        # read left image
        left_img = cv2.cvtColor(cv2.imread(left_image_filename), cv2.COLOR_BGR2RGB)
        left_img = cv2.resize(left_img, self.img_resolution, interpolation=cv2.INTER_LINEAR)
        left_img_resized = cv2.resize(left_img, self.img_size)
        left_img = np.transpose(left_img, (2,0,1))
        left_img_resized = np.transpose(left_img_resized, (2,0,1))

        # read right image
        right_img = cv2.cvtColor(cv2.imread(right_image_filename), cv2.COLOR_BGR2RGB)
        right_img = cv2.resize(right_img, self.img_resolution, interpolation=cv2.INTER_LINEAR)
        right_img_resized = cv2.resize(right_img, self.img_size)
        right_img = np.transpose(right_img, (2,0,1))
        right_img_resized = np.transpose(right_img_resized, (2,0,1))

        # read point-cloud
        velo_pc = read_velo_bin(velodyne_filename)
        # create mask
        velo_pc_mask = np.logical_and.reduce(((velo_pc[:,0] > self.xlim[0]), (velo_pc[:,0] < self.xlim[1]), \
                                              (velo_pc[:,1] > self.ylim[0]), (velo_pc[:,1] < self.ylim[1]), \
                                              (velo_pc[:,2] > self.zlim[0]), (velo_pc[:,2] < self.zlim[1])))

        # filter out
        velo_pc_filtered = velo_pc[velo_pc_mask]

        # convert point-cloud to volume
        velo_pc_vol = point_cloud_to_volume(velo_pc_filtered[:,:], vol_size=self.vol_size, xlim=self.xlim, ylim=self.ylim, zlim=self.zlim)

        # project points onto camera image plane
        projected_img = project_pc2image(velo_pc, self.T_lidar2cam, self.K, (left_img.shape[2], left_img.shape[1]))
        projected_img = cv2.resize(projected_img, self.img_size, interpolation=cv2.INTER_NEAREST)
        projected_img[projected_img > self.max_depth] = self.max_depth
        projected_img = projected_img[np.newaxis, ...]

        sample = {'cloud': velo_pc_filtered,                    \
                  'cloud_voxelized': velo_pc_vol,               \
                  'left_image': left_img,                       \
                  'left_image_resized': left_img_resized,       \
                  'right_image': right_img,                     \
                  'right_image_resized': right_img_resized,     \
                  'lidar_cam_projection': projected_img,        \
                  'label_dict': label_dict_list,                \
                  'label_vector':label_vector,                  \
                  'cloud_filename':self.velodyne_filenames[idx]}

        return sample