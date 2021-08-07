import os
import sys
import numpy as np
import math

# add path to sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.dirname(os.path.abspath('src/utils'))
sys.path.append(BASE_DIR)
sys.path.append(UTILS_DIR)

from utils import *

# function to write predictions and ground-truth to file in KITTI format for evaluations
# poses contains a list of [obj_class, conf, center_x, center_y, center_z, width, height, length, orientation]
def predictions2file(label_dict_list_all, txt_filenames, resolution, K, out_folder='results', exp='kitti', conf_thres=0.7, classes=['Car']):
    '''
    #Values    Name      Description
    ----------------------------------------------------------------------------
    1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                        'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                        'Misc' or 'DontCare'
    1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                        truncated refers to the object leaving image boundaries
    1    occluded     Integer (0,1,2,3) indicating occlusion state:
                        0 = fully visible, 1 = partly occluded
                        2 = largely occluded, 3 = unknown
    1    alpha        Observation angle of object, ranging [-pi..pi]
    4    bbox         2D bounding box of object in the image (0-based index):
                        contains left, top, right, bottom pixel coordinates
    3    dimensions   3D object dimensions: height, width, length (in meters)
    3    location     3D object location x,y,z in camera coordinates (in meters)
    1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
    1    score        Only for results: Float, indicating confidence in
                        detection, needed for p/r curves, higher is better.
    '''
    pred_folder = os.path.join(out_folder, exp)
    os.system('mkdir -p {}'.format(pred_folder))

    for idx in range(len(label_dict_list_all)):
        # extract poses
        label_dict_list = label_dict_list_all[idx]
        label_dict_to_write = []
        for label_dict in label_dict_list:
            if label_dict['conf'] > conf_thres: label_dict_to_write.append(label_dict) 

        # write predictions to file
        out_fname = os.path.join(pred_folder, txt_filenames[idx])
        with open(out_fname, 'w') as f:

            for label_dict in label_dict_to_write:
                obj_class, conf, x, y, z, w, h, l, orientation = label_dict['class'], \
                                                                 label_dict['conf'], \
                                                                 label_dict['x'], \
                                                                 label_dict['y'], \
                                                                 label_dict['z'], \
                                                                 label_dict['w'], \
                                                                 label_dict['h'], \
                                                                 label_dict['l'], \
                                                                 label_dict['yaw']

                # check if this object evaluation was requiested
                if obj_class in classes:
                
                    # convert to camera coordinate system
                    x_cam = -y
                    y_cam = -z
                    z_cam = x

                    # build kitti format labels
                    obj_type = obj_class
                    truncated = 0
                    occluded = 3
                    alpha = 0.0
                    y_proj = y_cam
                    y_cam = y_cam + (h / 2.0)
                    rotation_y = np.pi - orientation
                    if rotation_y < -np.pi: rotation_y += np.pi
                    elif rotation_y > np.pi: rotation_y -= np.pi
                    # extract vertices of bboxes
                    pts_3d = []

                    ## get 2D bounding-box
                    # front 4 vertices
                    pts_3d.append([x_cam-w/2., y_proj-h/2., z_cam-l/2.])
                    pts_3d.append([x_cam+w/2., y_proj-h/2., z_cam-l/2.])
                    pts_3d.append([x_cam+w/2., y_proj+h/2., z_cam-l/2.])
                    pts_3d.append([x_cam-w/2., y_proj+h/2., z_cam-l/2.])
                    # vertices behind
                    pts_3d.append([x_cam-w/2., y_proj-h/2., z_cam+l/2.])
                    pts_3d.append([x_cam+w/2., y_proj-h/2., z_cam+l/2.])
                    pts_3d.append([x_cam+w/2., y_proj+h/2., z_cam+l/2.])
                    pts_3d.append([x_cam-w/2., y_proj+h/2., z_cam+l/2.])

                    # change the orientation so that 0 degrees is aligned with z-axis
                    orientation = math.degrees(orientation-np.pi)

                    # move the bbox to the origin and then rotate as given orientation
                    for i in range(len(pts_3d)):
                        # move the center of bbox to origin
                        pts_3d[i] = affineTransform(pts_3d[i], 0, 0, 0, -x_cam, -y_proj, -z_cam)
                        # rotate points and move the bbox back to x,y,z
                        pts_3d[i] = affineTransform(pts_3d[i], 0, orientation, 0, x_cam, y_proj, z_cam)

                    # get 2d projection
                    pts_2d = np.array(pts_3d_to_2d(pts_3d, K, convert2int=False))

                    # 2d bbox
                    x_min = min(pts_2d[:,0])
                    x_max = max(pts_2d[:,0])
                    y_min = min(pts_2d[:,1])
                    y_max = max(pts_2d[:,1])

                    # check for boundary
                    if(x_min < 0):
                        x_min = 0.0
                    elif(x_min >= resolution[0]):
                        x_max = resolution[0] - 1
                    
                    if(y_min < 0):
                        y_min = 0.0
                    elif(y_min >= resolution[1]):
                        y_max = resolution[1] - 1

                    # make sure the min and max makes sense
                    if ((x_max - x_min) <= 0) or ((y_max - y_min) <= 0):
                        continue

                    # kitti label
                    kitti_pred_label = '{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(
                        obj_type, truncated, occluded, alpha, x_min, y_min, x_max, y_max, h, w, l, x_cam, y_cam, z_cam, rotation_y, conf)
                    print(kitti_pred_label, file=f)

# compute depth metrics
def compute_depth_metrics(gt, pred, max_depth=70.0):
    # create a mask for gt==0
    mask = (gt==0) | (gt > max_depth)

    # mask the ground-truth and prediction
    gt = np.ma.masked_array(gt, mask=mask)
    pred = np.ma.masked_array(pred, mask=mask)
    
    thresh = np.ma.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.ma.sqrt(rmse.mean())

    rmse_log = (np.ma.log(gt) - np.ma.log(pred)) ** 2
    rmse_log = np.ma.sqrt(rmse_log.mean())

    abs_rel = np.ma.mean(np.abs(gt - pred) / gt)

    sq_rel = np.ma.mean(((gt - pred)**2) / gt)

    return {'abs_rel':abs_rel, 'sq_rel':sq_rel, 'rmse':rmse, 'rmse_log':rmse_log, 'a1':a1, 'a2':a2, 'a3':a3}