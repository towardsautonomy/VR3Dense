import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from vr3d_utils import *

# file path
velo_file = '/media/shubham/GoldMine/datasets/KITTI/raw/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/0000000000.bin'

# load point-cloud
velo_pc = read_velo_bin(velo_file)

# minimum and maximum limits
xlim = [-50, 50]
ylim = [-50, 50]
zlim = [-10, 10]

# create mask
velo_pc_mask = np.logical_and.reduce(((velo_pc[:,0] > xlim[0]), (velo_pc[:,0] < xlim[1]), \
                                      (velo_pc[:,1] > ylim[0]), (velo_pc[:,1] < ylim[1]), \
                                      (velo_pc[:,2] > zlim[0]), (velo_pc[:,2] < zlim[1])))

# filter out
velo_pc = velo_pc[velo_pc_mask]

'''
## convert to open3d type
# Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(velo_pc)
o3d.visualization.draw_geometries([pcd])
'''

# convert point-cloud to volume
velo_pc_vol = point_cloud_to_volume(velo_pc[:,:], vol_size=(256,256,16), \
                                        xlim=xlim, ylim=ylim, zlim=zlim

# convert back to point-cloud
velo_pc_recon = volume_to_point_cloud(velo_pc_vol)

## convert to open3d type
# Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
pcd_recon = o3d.geometry.PointCloud()
pcd_recon.points = o3d.utility.Vector3dVector(velo_pc_recon)
o3d.visualization.draw_geometries([pcd_recon])