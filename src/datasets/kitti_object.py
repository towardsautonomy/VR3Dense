# import packages
import os
import sys
import torch
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
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

    def __init__(self, dataroot, n_xgrids, n_ygrids, 
                       xlim=(0.0, 70.0), ylim=(-40.0, 40.0), zlim=(-2.5, 1.0),
                       vol_size=(512,512,32), img_size=(256,256)):
        """
        Args:
            dataroot (string): Root Directory of KITTI object dataset.
        """
        # paths
        self.dataroot = dataroot
        self.train_dir = os.path.join(dataroot, 'training')
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

    # method to get each item
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        left_image_filename = self.left_image_filenames[idx]
        right_image_filename = self.right_image_filenames[idx]
        velodyne_filename = self.velodyne_filenames[idx]
        label_dict_list = self.label_dict_list[idx]
        label_vector = build_label_vector(label_dict_list, self.n_xgrids, self.n_ygrids, \
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
        projected_img[projected_img > self.xlim[1]] = self.xlim[1]
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

# main function
if __name__ == '__main__':
    from utils import *

    dataset = KITTIObjectDataset('/media/shubham/GoldMine/datasets/KITTI/object', \
                                 n_xgrids=32, n_ygrids=32)

    # visualization window
    cv2.namedWindow('VR3Dense', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('VR3Dense', 900, 1440)
    # show 100 samples
    for i in range(100): 
        sample = dataset[i]
        # build label vector and then recover poses and classes
        label_vec = build_label_vector(sample['label_dict'], dataset.n_xgrids, dataset.n_ygrids)
        label_dict = decompose_label_vector(label_vec, dataset.n_xgrids, dataset.n_ygrids)
        # recover point-cloud from voxelized volume
        pc_voxelized = volume_to_point_cloud(sample['cloud_voxelized'], vol_size=(512,512,32), xlim=dataset.xlim, ylim=dataset.ylim, zlim=dataset.zlim)

        # draw point-cloud
        canvasSize = 1200
        pc_bbox_img = draw_point_cloud_w_bbox(pc_voxelized, label_dict, canvasSize=canvasSize)

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
        projected_pts = colorize_depth_map(projected_pts)
        projected_pts = cv2.resize(projected_pts, (width, height), interpolation = cv2.INTER_NEAREST) 
        projected_pts = cv2.cvtColor(projected_pts, cv2.COLOR_RGB2BGR)
        projected_pts = np.array(projected_pts, dtype=np.uint8)
        projected_pts = draw_bbox_img(projected_pts, label_cam, dataset.K)

        # concat image with point-cloud 
        img_viz = cv2.vconcat([img_bgr, projected_pts, pc_bbox_img_bgr])

        cv2.imshow('VR3Dense', img_viz)
        cv2.waitKey(0)