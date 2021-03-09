import glob
import numpy as np
import os
import shutil

n_test_samples = 100
kitti_path = '/media/shubham/GoldMine/datasets/KITTI/object/testing/'
kitti_val_dst_path = '/media/shubham/GoldMine/datasets/KITTI/object_subset/testing/'
kitti_val_img2_path = kitti_val_dst_path + 'image_2/'
kitti_val_img3_path = kitti_val_dst_path + 'image_3/'
kitti_val_velo_path = kitti_val_dst_path + 'velodyne/'
kitti_val_label_path = kitti_val_dst_path + 'label_2/'

# create directory
os.system('mkdir -p {}'.format(kitti_val_img2_path))
os.system('mkdir -p {}'.format(kitti_val_img3_path))
os.system('mkdir -p {}'.format(kitti_val_velo_path))
os.system('mkdir -p {}'.format(kitti_val_label_path))

# label files
txt_files = sorted(glob.glob(kitti_path+'label_2/*.txt'))

# shuffle the data
p = np.random.permutation(len(txt_files))
txt_files = np.array(txt_files)[p]
txt_files = txt_files[:min(n_test_samples, len(txt_files))]

# get corresponding img files
img_files = []
for idx, txt_file in enumerate(txt_files):
    img2_file = kitti_path+'image_2/'+txt_file[-10:-4]+'.png'
    img3_file = kitti_path+'image_3/'+txt_file[-10:-4]+'.png'
    velo_file = kitti_path+'velodyne/'+txt_file[-10:-4]+'.bin'

    if os.path.exists(txt_file) and os.path.exists(img2_file):
        # destination file names
        dst_txt_file = kitti_val_label_path + str(idx).zfill(6) + '.txt'
        dst_img2_file = kitti_val_img2_path + str(idx).zfill(6) + '.png'
        dst_img3_file = kitti_val_img3_path + str(idx).zfill(6) + '.png'
        dst_velo_file = kitti_val_velo_path + str(idx).zfill(6) + '.bin'
        # move these files
        shutil.copy(txt_file, dst_txt_file)
        shutil.copy(img2_file, dst_img2_file)
        shutil.copy(img3_file, dst_img3_file)
        shutil.copy(velo_file, dst_velo_file)

        print('Copied files:')
        print('{} -> {}'.format(txt_file, dst_txt_file))