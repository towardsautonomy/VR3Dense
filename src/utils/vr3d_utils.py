import os
from eulerangles import euler2mat
import numpy as np
import math
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.cm as cm
from google_drive_downloader import GoogleDriveDownloader
from affine_transform import affineTransform

# label vector
# label_map = {'Car':0} 
# label_map = {'Car':0, 'Van':1, 'Truck':2, 'Cyclist':3, 'Pedestrian':4} 
label_map = {'Car':0, 'Cyclist':1, 'Pedestrian':2} 
pose_fields = ['conf','x','y','z','l','w','h','cos_yaw','sin_yaw']
pose_vec_len = len(pose_fields)

# pretrained weights
pretrained_weights = [{'exp':'vr3d.learning_rate_0.0001.n_xgrids_16.n_ygrids_16.xlim_0.0_70.0.ylim_-25.0_25.0.zlim_-2.5_1.0.max_depth_100.0.vol_size_256x256x16.img_size_512x256.dense_depth_True.concat_latent_vector_True.exp_id_kitti',
                      'url':'https://drive.google.com/file/d/13JVrBhcLDEDSMkfsnAn7Iuu2Sma1H-iz/view?usp=sharing',
                      'file_id':'13JVrBhcLDEDSMkfsnAn7Iuu2Sma1H-iz'}
                     ]

# load pretrained weights
def load_pretrained_weights(model, modeldir, exp_str):
    # best checkpoint model name
    model_exp_dir = os.path.join(modeldir, exp_str)
    best_ckpt_model = os.path.join(model_exp_dir, 'checkpoint_best.pt')
    # check if the model exists
    if os.path.exists(best_ckpt_model):
        model.load_state_dict(torch.load(best_ckpt_model, map_location=lambda storage, loc: storage))
        print('Loaded pre-trained weights: {}'.format(best_ckpt_model))
    else:
        found = False
        print('Pre-trained weights not found. Attempting to download.')
        for pretrained_weight in pretrained_weights:
            for key in pretrained_weight.keys():
                if key == 'exp':
                    if exp_str == pretrained_weight[key]:
                        os.system('mkdir -p {}'.format(model_exp_dir))
                        GoogleDriveDownloader.download_file_from_google_drive(file_id=pretrained_weight['file_id'],
                                    dest_path=os.path.join(model_exp_dir, 'checkpoint_best.pt'),
                                    unzip=False,
                                    showsize=True)

                        model.load_state_dict(torch.load(best_ckpt_model, map_location=lambda storage, loc: storage))
                        print('Loaded pre-trained weights: {}'.format(best_ckpt_model))
                        found = True

        if found == False:
            print('Unable to find pretrained weights with this experiment configuration')
            raise Exception('Pre-trained weights not found.')

    return model

# this function returns label index given a label string
def label_to_idx(label):
    if label in label_map.keys():
        return label_map[label]
    else:
        raise('Unsupported object class')

# this function returns index given a label
def idx_to_label(idx):
    label_ret = 'DontCare'
    for label in label_map.items():
        if idx == label[1]:
            label_ret = label[0]

    if label_ret == 'DontCare':
        raise('Unsupported object class')

    return label_ret

## Objects
# normalize functions
def normalize_l(l):
    return l / 10.0
def normalize_w(w):
    return w / 5.0
def normalize_h(h):
    return h / 5.0

# denormalize functions
def denormalize_l(l):
    return l * 10.0
def denormalize_w(w):
    return w * 5.0
def denormalize_h(h):
    return h * 5.0

## Image and Depth
# normalize
def normalize_img(img):
    return (img / 255.0) - 0.5

# denormalize
def denormalize_img(img):
    return (img + 0.5) * 255.0

# normalize
def normalize_depth(depth, max_depth):
    return (depth * 2.0 / max_depth) - 1.0

# denormalize
def denormalize_depth(depth, max_depth):
    return (depth + 1.0) * (max_depth / 2)

# draw point-cloud
def draw_point_cloud_topdown(input_points, canvasSize=800, radius=1,
                             zrot=0, switch_xyz=[0,1,2], 
                             xlim=(0.0,70.0), ylim=(-50.0,50.0), zlim=(-5.0,10.0), background_color=(255,255,255)):
    """ Render point cloud to image with alpha channel.
        Input:
            points: Nx3 numpy array (+z is up direction)
        Output:
            colorized image as numpy array of size canvasSizexcanvasSize
    """
    # create mask
    input_points_mask = np.logical_and.reduce(((input_points[:,0] > xlim[0]), (input_points[:,0] < xlim[1]), \
                                               (input_points[:,1] > ylim[0]), (input_points[:,1] < ylim[1]), \
                                               (input_points[:,2] > zlim[0]), (input_points[:,2] < zlim[1])))

    # filter out
    input_points = input_points[input_points_mask]
    # hue range
    huerange = (240, 0) # green to red

    # image plane
    image = np.zeros((canvasSize, canvasSize, 3), dtype=np.uint8)
    if input_points is None or input_points.shape[0] == 0:
        return image

    # x in point-cloud to image y coordinate
    def pcx_to_imgy(pcx):
        m2px = (xlim[1]-xlim[0]) / float(canvasSize)
        imgy = canvasSize - int(pcx / m2px)
        return imgy

    # y in point-cloud to image x coordinate
    def pcy_to_imgx(pcy):
        m2px = (ylim[1]-ylim[0]) / float(canvasSize)
        imgx = int((float(canvasSize) / 2.0) - (pcy / m2px))
        return imgx

    points = input_points
    M = euler2mat(zrot, 0, 0)
    points = (np.dot(M, points.transpose())).transpose()

    # go through each point
    for pt in points:
        imgx = pcy_to_imgx(pt[1])
        imgy = pcx_to_imgy(pt[0])

        # draw circle
        ztohue = (zlim[1] - zlim[0]) / (huerange[0] - huerange[1])
        hue = huerange[0] - int((pt[2] - zlim[0]) / ztohue)
        cv2.circle(image, (imgx, imgy), radius=radius, color=(hue,255,128), thickness=-1)

    # convert to BGR
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    image[np.where(np.all(image == [0,0,0], axis=-1))] = background_color

    # scale between 0.0 and 1.0
    image = image.astype(np.float32) / 255.0
    return image

# draw point-cloud with bounding-box
def draw_point_cloud_w_bbox(input_points, label_dict_list, canvasSize=800, radius=1,
                             xlim=(0.0, 70.0), ylim=(-50.0,50.0), zlim=(-5.0,10.0), background_color=(255,255,255), text_color=(0,0,0)):
    """ Render point cloud to image and draw bounding box.
        Input:
            input_points: Nx3 numpy array (+z is up direction)
        Output:
            colorized image as numpy array of size canvasSizexcanvasSize
    """
    # create mask
    input_points_mask = np.logical_and.reduce(((input_points[:,0] > xlim[0]), (input_points[:,0] < xlim[1]), \
                                               (input_points[:,1] > ylim[0]), (input_points[:,1] < ylim[1]), \
                                               (input_points[:,2] > zlim[0]), (input_points[:,2] < zlim[1])))

    # filter out
    input_points = input_points[input_points_mask]
    # hue range
    huerange = (240, 0) # green to red

    # image plane
    image = np.zeros((canvasSize, canvasSize, 3), dtype=np.uint8)
    if input_points is None or input_points.shape[0] == 0:
        return image

    # x in point-cloud to image y coordinate
    def pcx_to_imgy(pcx):
        m2px = (xlim[1]-xlim[0]) / float(canvasSize)
        imgy = canvasSize - int(pcx / m2px)
        return imgy

    # y in point-cloud to image x coordinate
    def pcy_to_imgx(pcy):
        m2px = (ylim[1]-ylim[0]) / float(canvasSize)
        imgx = int((float(canvasSize) / 2.0) - (pcy / m2px))
        return imgx

    # go through each point
    for pt in input_points:
        imgx = pcy_to_imgx(pt[1])
        imgy = pcx_to_imgy(pt[0])

        # draw circle
        ztohue = (zlim[1] - zlim[0]) / (huerange[0] - huerange[1])
        hue = huerange[0] - int((pt[2] - zlim[0]) / ztohue)
        cv2.circle(image, (imgx, imgy), radius=radius, color=(hue,255,128), thickness=-1)

    # convert to BGR
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    image[np.where(np.all(image == [0,0,0], axis=-1))] = background_color

    # draw bounding boxes
    for label_dict in label_dict_list:
        x,y,z,l,w,h,yaw = label_dict['x'],label_dict['y'],label_dict['z'],label_dict['l'],label_dict['w'],label_dict['h'],label_dict['yaw']
        yaw = np.pi/2.0 + yaw

        # compute corners in birds-eye view
        corners = []
        corners.append([ l/2, -w/2, 0.0])
        corners.append([ l/2,  w/2, 0.0])
        corners.append([-l/2,  w/2, 0.0])
        corners.append([-l/2, -w/2, 0.0])
        corners = np.asarray(corners, dtype=np.float32)

        # rotate input_points
        M = euler2mat(yaw, 0, 0)
        corners = (np.dot(M, corners.transpose())).transpose()
        corners = corners + [x, y, z]

        # corners in pixel
        corners_px = []
        for corner in corners:
            corners_px.append([pcy_to_imgx(corner[1]), pcx_to_imgy(corner[0])])
        corners_px = np.asarray(corners_px, dtype=np.int)

        # draw bounding box
        cv2.line(image, (corners_px[0,0], corners_px[0,1]), (corners_px[1,0], corners_px[1,1]), color=(0,0,0), thickness=6)
        cv2.line(image, (corners_px[0,0], corners_px[0,1]), (corners_px[1,0], corners_px[1,1]), color=(255,0,0), thickness=2)
        cv2.line(image, (corners_px[1,0], corners_px[1,1]), (corners_px[2,0], corners_px[2,1]), color=(255,0,0), thickness=2)
        cv2.line(image, (corners_px[2,0], corners_px[2,1]), (corners_px[3,0], corners_px[3,1]), color=(255,0,0), thickness=2)
        cv2.line(image, (corners_px[3,0], corners_px[3,1]), (corners_px[0,0], corners_px[0,1]), color=(255,0,0), thickness=2)

        # # get top-left coordinates
        # tl = (np.min(corners_px[:,0]), np.min(corners_px[:,1]))

        # # write class
        # cv2.putText(image, '{} {:.1f}%'.format(label_dict['class'], label_dict['conf']*100.0), 
        #     (tl[0],tl[1]-5), 
        #     cv2.FONT_HERSHEY_SIMPLEX, 
        #     0.5,
        #     color=text_color,
        #     thickness=1,
        #     lineType=2)

    # scale between 0.0 and 1.0
    image = image.astype(np.float32) / 255.0
    return image

# draw point-cloud with bounding-box
def draw_point_cloud_w_bbox_id(input_points, label_dict_list, canvasSize=800, radius=1,
                             xlim=(0.0, 70.0), ylim=(-50.0,50.0), zlim=(-5.0,10.0), background_color=(255,255,255), text_color=(0,0,0)):
    """ Render point cloud to image and draw bounding box.
        Input:
            input_points: Nx3 numpy array (+z is up direction)
        Output:
            colorized image as numpy array of size canvasSizexcanvasSize
    """
    # create mask
    input_points_mask = np.logical_and.reduce(((input_points[:,0] > xlim[0]), (input_points[:,0] < xlim[1]), \
                                               (input_points[:,1] > ylim[0]), (input_points[:,1] < ylim[1]), \
                                               (input_points[:,2] > zlim[0]), (input_points[:,2] < zlim[1])))

    # filter out
    input_points = input_points[input_points_mask]
    # hue range
    huerange = (240, 0) # green to red

    # image plane
    image = np.zeros((canvasSize, canvasSize, 3), dtype=np.uint8)
    if input_points is None or input_points.shape[0] == 0:
        return image

    # x in point-cloud to image y coordinate
    def pcx_to_imgy(pcx):
        m2px = (xlim[1]-xlim[0]) / float(canvasSize)
        imgy = canvasSize - int(pcx / m2px)
        return imgy

    # y in point-cloud to image x coordinate
    def pcy_to_imgx(pcy):
        m2px = (ylim[1]-ylim[0]) / float(canvasSize)
        imgx = int((float(canvasSize) / 2.0) - (pcy / m2px))
        return imgx

    # go through each point
    for pt in input_points:
        imgx = pcy_to_imgx(pt[1])
        imgy = pcx_to_imgy(pt[0])

        # draw circle
        ztohue = (zlim[1] - zlim[0]) / (huerange[0] - huerange[1])
        hue = huerange[0] - int((pt[2] - zlim[0]) / ztohue)
        cv2.circle(image, (imgx, imgy), radius=radius, color=(hue,255,255), thickness=-1)

    # convert to BGR
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    image[np.where(np.all(image == [0,0,0], axis=-1))] = background_color

    # draw bounding boxes
    for label_dict in label_dict_list:
        x,y,z,l,w,h,yaw = label_dict['x'],label_dict['y'],label_dict['z'],label_dict['l'],label_dict['w'],label_dict['h'],label_dict['yaw']
        yaw = np.pi/2.0 + yaw

        # compute corners in birds-eye view
        corners = []
        corners.append([ l/2, -w/2, 0.0])
        corners.append([ l/2,  w/2, 0.0])
        corners.append([-l/2,  w/2, 0.0])
        corners.append([-l/2, -w/2, 0.0])
        corners = np.asarray(corners, dtype=np.float32)

        # rotate input_points
        M = euler2mat(yaw, 0, 0)
        corners = (np.dot(M, corners.transpose())).transpose()
        corners = corners + [x, y, z]

        # corners in pixel
        corners_px = []
        for corner in corners:
            corners_px.append([pcy_to_imgx(corner[1]), pcx_to_imgy(corner[0])])
        corners_px = np.asarray(corners_px, dtype=np.int)

        # draw bounding box
        cv2.line(image, (corners_px[0,0], corners_px[0,1]), (corners_px[1,0], corners_px[1,1]), color=(0,0,0), thickness=6)
        cv2.line(image, (corners_px[0,0], corners_px[0,1]), (corners_px[1,0], corners_px[1,1]), color=(255,0,0), thickness=2)
        cv2.line(image, (corners_px[1,0], corners_px[1,1]), (corners_px[2,0], corners_px[2,1]), color=(255,0,0), thickness=2)
        cv2.line(image, (corners_px[2,0], corners_px[2,1]), (corners_px[3,0], corners_px[3,1]), color=(255,0,0), thickness=2)
        cv2.line(image, (corners_px[3,0], corners_px[3,1]), (corners_px[0,0], corners_px[0,1]), color=(255,0,0), thickness=2)

        # get top-left coordinates
        tl = (np.min(corners_px[:,0]), np.min(corners_px[:,1]))

        # write object id and class
        cv2.putText(image, '{} {:.1f}% | id {}'.format(label_dict['class'], label_dict['conf']*100.0, label_dict['id']), 
            (tl[0],tl[1]-5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6,
            color=text_color,
            thickness=2,
            lineType=2)

    # scale between 0.0 and 1.0
    image = image.astype(np.float32) / 255.0
    return image

def draw_bbox_img(img, label_dict_cam, K):
    """ Draw bounding-box on image using label dictionary
    in the camera coordinate system and camera intrinsic matrix, K
    """
    img_copy = img.copy()

    # colors
    WHITE   = (255,255,255)
    RED     = (255,0,0)
    YELLOW  = (255,255,0)
    GREEN   = (0,255,0)
    CYAN    = (0,255,255)
    BLUE    = (0,0,255)

    # draw bounding boxes
    for label in label_dict_cam:
        x,y,z,l,w,h,yaw = label['x'],label['y'],label['z'],label['l'],label['w'],label['h'],label['yaw']
        # yaw = np.pi/2.0 + yaw
        yaw = (np.pi/2.0) - yaw

        # extract vertices of bboxes
        pts_3d = []
        # front 4 vertices
        pts_3d.append([x-w/2., y-h/2., z-l/2.])
        pts_3d.append([x+w/2., y-h/2., z-l/2.])
        pts_3d.append([x+w/2., y+h/2., z-l/2.])
        pts_3d.append([x-w/2., y+h/2., z-l/2.])
        # vertices behind
        pts_3d.append([x-w/2., y-h/2., z+l/2.])
        pts_3d.append([x+w/2., y-h/2., z+l/2.])
        pts_3d.append([x+w/2., y+h/2., z+l/2.])
        pts_3d.append([x-w/2., y+h/2., z+l/2.])

        # # change the orientation so that 0 degrees is aligned with z-axis
        yaw = math.degrees(yaw)

        # move the bbox to the origin and then rotate as given orientation
        for i in range(len(pts_3d)):
            # move the center of bbox to origin
            pts_3d[i] = affineTransform(pts_3d[i], 0, 0, 0, -x, -y, -z)
            # rotate points and move the bbox back to x,y,z
            pts_3d[i] = affineTransform(pts_3d[i], 0, yaw, 0, x, y, z)

        # get 2d projection
        pts_2d = pts_3d_to_2d(pts_3d, K)

        # draw front rectangle
        for i in range(3):
            cv2.line(img, (pts_2d[i][0], pts_2d[i][1]), (pts_2d[i+1][0], pts_2d[i+1][1]), color=WHITE, thickness=2)
        cv2.line(img, (pts_2d[3][0], pts_2d[3][1]), (pts_2d[0][0], pts_2d[0][1]), color=WHITE, thickness=2)

        # front cross
        cv2.line(img, (pts_2d[0][0], pts_2d[0][1]), (pts_2d[2][0], pts_2d[2][1]), color=WHITE, thickness=2)
        cv2.line(img, (pts_2d[1][0], pts_2d[1][1]), (pts_2d[3][0], pts_2d[3][1]), color=WHITE, thickness=2)

        # draw back rectangle
        for i in range(4,7):
            cv2.line(img, (pts_2d[i][0], pts_2d[i][1]), (pts_2d[i+1][0], pts_2d[i+1][1]), color=RED, thickness=2)
        cv2.line(img, (pts_2d[7][0], pts_2d[7][1]), (pts_2d[4][0], pts_2d[4][1]), color=RED, thickness=2)

        # connecting two rectangles
        cv2.line(img, (pts_2d[0][0], pts_2d[0][1]), (pts_2d[4][0], pts_2d[4][1]), color=RED, thickness=2)
        cv2.line(img, (pts_2d[1][0], pts_2d[1][1]), (pts_2d[5][0], pts_2d[5][1]), color=RED, thickness=2)
        cv2.line(img, (pts_2d[2][0], pts_2d[2][1]), (pts_2d[6][0], pts_2d[6][1]), color=RED, thickness=2)
        cv2.line(img, (pts_2d[3][0], pts_2d[3][1]), (pts_2d[7][0], pts_2d[7][1]), color=RED, thickness=2)

        # bottom face
        # cv2.fillConvexPoly(img_copy, np.array([pts_2d[2,:], pts_2d[3,:], pts_2d[7,:], pts_2d[6,:]], dtype=np.int32), color=BLUE)

        # # get minimum x and y
        # x_min = min(pts_2d[:,0])
        # y_min = min(pts_2d[:,1])

        # # a rectangle behind text
        # cv2.rectangle(img, (x_min,y_min-20), (x_min+150,y_min), color=(0,0,0), thickness=-1)
        # cv2.rectangle(img_copy, (x_min,y_min-20), (x_min+150,y_min), color=(0,0,0), thickness=-1)

        # # write object class
        # cv2.putText(img, label['class']+' {:.1f}%'.format(label['conf']*100.0), 
        #     (x_min+5, y_min-5), 
        #     cv2.FONT_HERSHEY_COMPLEX, 
        #     0.6,
        #     color=(255,255,255),
        #     thickness=1,
        #     lineType=2)

    # return image
    # img_ret = cv2.addWeighted(img, 1.0, img_copy, 0.5, 0.)
    return img

def build_label_vector(label_dict_list, n_xgrids, n_ygrids,
                        xlim=(0.0, 70.0), ylim=(-50.0,50.0), zlim=(-10.0,10.0)):
    """ Build the ground-truth label vector 
        given a set of poses, classes, and 
        number of grids.
        Input:
            label_dict_list: list of label dictionary
            n_xgrids: number of grids in the x direction
            n_ygrids: number of grids in the y direction
        Output:
            label vector
    """
    obj_label_len = pose_vec_len + len(label_map) # 9 for poses, rest for object classes
    label_vector = np.zeros((n_xgrids * n_ygrids * obj_label_len), dtype=np.float32)

    # iterate through each pose
    for i, label_dict in enumerate(label_dict_list):
        x,y,z,l,w,h,yaw = label_dict['x'],label_dict['y'],label_dict['z'],label_dict['l'],label_dict['w'],label_dict['h'],label_dict['yaw']

        # obtain x index
        xstop = (xlim[1] - xlim[0]) / float(n_xgrids)
        x_idx = math.floor((x - xlim[0]) / xstop)

        # obtain y index
        ystop = (ylim[1] - ylim[0]) / float(n_ygrids)
        y_idx = math.floor((y - ylim[0]) / ystop)

        # pose vector
        x_norm = ((x - xlim[0]) - (x_idx * xstop)) / xstop
        y_norm = ((y - ylim[0]) - (y_idx * ystop)) / ystop
        z_norm = (z - zlim[0]) / (zlim[1] - zlim[0])
        l_norm = normalize_l(l)
        w_norm = normalize_w(w)
        h_norm = normalize_h(h)
        cos_yaw_norm = (np.cos(yaw) + 1.0) / 2.0
        sin_yaw_norm = (np.sin(yaw) + 1.0) / 2.0
        # yaw_norm = (yaw + np.pi) / (2 * np.pi)
        pose_vec = [1.0, x_norm, y_norm, z_norm, l_norm, w_norm, h_norm, cos_yaw_norm, sin_yaw_norm]

        # class vector
        class_vec = [0.0]*len(label_map)
        class_idx = label_to_idx(label_dict['class'])
        class_vec[class_idx] = 1.0

        # label vector for this object
        label_vec_this_obj = pose_vec + class_vec

        # label index
        label_idx = ((x_idx * n_ygrids) + y_idx) * obj_label_len

        # populate label vector
        label_vector[label_idx:label_idx+obj_label_len] = label_vec_this_obj
    
    # return the label vector
    return label_vector

def decompose_label_vector(label_vector, n_xgrids, n_ygrids,
                            xlim=(0.0, 70.0), ylim=(-50.0,50.0), zlim=(-10.0,10.0),
                            conf_thres=0.7, nms=True, iou_thres=0.1):
    """ Build the ground-truth label vector 
        given a set of poses, classes, and 
        number of grids.
        Input:
            label_vector: label vector outputted from the model
            n_xgrids: number of grids in the x direction
            n_ygrids: number of grids in the y direction
        Output:
            poses: list of object poses [x,y,z,l,w,h,yaw]
            classes: list of object classes
    """
    conf = []
    poses = []
    classes = []
    label_dict_list = []

    # obtain x index
    xstop = (xlim[1] - xlim[0]) / float(n_xgrids)

    # obtain y index
    ystop = (ylim[1] - ylim[0]) / float(n_ygrids)

    # length of each object label
    obj_label_len = pose_vec_len + len(label_map) # 8 for poses, rest for object classes

    # reshape the vector
    label_vector_reshaped = np.reshape(label_vector, (-1, obj_label_len))

    # get each element
    obj_confidences = label_vector_reshaped[:, 0]
    obj_poses = label_vector_reshaped[:, 1:pose_vec_len]
    obj_class_one_hot = label_vector_reshaped[:, pose_vec_len:]

    # iterate through each element
    for i, obj_conf in enumerate(obj_confidences):
        if obj_conf > conf_thres:
            # pose vector
            x_norm, y_norm, z_norm, l_norm, w_norm, h_norm, cos_yaw_norm, sin_yaw_norm = obj_poses[i]

            # get indices
            x_idx = math.floor(i / n_xgrids)
            y_idx = i - (x_idx * n_xgrids)

            # denormalize pose
            x = (x_norm * xstop) + (x_idx * xstop) + xlim[0]
            y = (y_norm * ystop) + (y_idx * ystop) + ylim[0]
            z = (z_norm * (zlim[1] - zlim[0])) + zlim[0]
            l = denormalize_l(l_norm)
            w = denormalize_w(w_norm)
            h = denormalize_h(h_norm)
            cos_yaw = (cos_yaw_norm * 2.0) - 1.0
            sin_yaw = (sin_yaw_norm * 2.0) - 1.0
            yaw = np.arctan2(sin_yaw, cos_yaw)

            # add poses, classes, and conf
            label_dict = {}
            label_dict['conf'] = obj_conf
            label_dict['x'] = x
            label_dict['y'] = y
            label_dict['z'] = z
            label_dict['l'] = l
            label_dict['w'] = w
            label_dict['h'] = h
            label_dict['yaw'] = yaw
            label_dict['class'] = idx_to_label(np.argmax(obj_class_one_hot[i]))
            label_dict_list.append(label_dict)
    
    # non-max suppression
    if nms == True:
        label_dict_list = non_max_suppression(label_dict_list, iou_threshold=iou_thres)

    # return label dictionary
    return label_dict_list

def point_cloud_to_volume(points, vol_size=(256,256,16), \
                            xlim=(0.0, 70.0), ylim=(-50.0,50.0), zlim=(-10.0,10.0)):
    """ input is Nx3 points.
        output is vol_size
    """
    vol = np.zeros(vol_size)
    voxel_size = ((xlim[1]-xlim[0])/float(vol_size[0]), \
                  (ylim[1]-ylim[0])/float(vol_size[1]), \
                  (zlim[1]-zlim[0])/float(vol_size[2]))

    locations = (points - (xlim[0],ylim[0],zlim[0])) / voxel_size
    locations = locations.astype(int)
    vol[locations[:,0],locations[:,1],locations[:,2]] += 1.0

    # change the axis for pytorch
    vol = np.transpose(vol, (2, 0, 1))
    # return volume
    return vol

def volume_to_point_cloud(vol, vol_size=(256,256,16), \
                            xlim=(0.0, 70.0), ylim=(-50.0,50.0), zlim=(-10.0,10.0)):
    """ vol is occupancy grid (value = 0 or 1) of size vol_size
        return Nx3 numpy array.
    """
    # change the axis for pytorch
    vol = np.transpose(vol, (1, 2, 0))

    assert((vol.shape[0] == vol_size[0]) and \
           (vol.shape[1] == vol_size[1]) and \
           (vol.shape[2] == vol_size[2]))
    voxel_size = ((xlim[1]-xlim[0])/float(vol_size[0]), \
                  (ylim[1]-ylim[0])/float(vol_size[1]), \
                  (zlim[1]-zlim[0])/float(vol_size[2]))
    points = []
    for a in range(vol_size[0]):
        for b in range(vol_size[1]):
            for c in range(vol_size[2]):
                if vol[a,b,c] > 0:
                    point = np.array([a,b,c])*voxel_size + (xlim[0],ylim[0],zlim[0])
                    points.append(np.array(point))

    if len(points) == 0:
        return np.zeros((0,3))
    points = np.vstack(points)
    return points

def read_velo_bin(filename):
    """" read velodyne point-clouds from .bin files. """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points = points[:, :3]  # exclude luminance
    return points

# intersection-over-union
def iou(labelA, labelB):
    poseA = [labelA['x'],labelA['y'],labelA['z'],labelA['l'],labelA['w'],labelA['h'],labelA['yaw']]
    poseB = [labelB['x'],labelB['y'],labelB['z'],labelB['l'],labelB['w'],labelB['h'],labelB['yaw']]
    # get min and max coordinates
    xmin = max(poseA[0]-(poseA[3]/2.0), poseB[0]-(poseB[3]/2.0))
    ymin = max(poseA[1]-(poseA[4]/2.0), poseB[1]-(poseB[4]/2.0))
    xmax = min(poseA[0]+(poseA[3]/2.0), poseB[0]+(poseB[3]/2.0))
    ymax = min(poseA[1]+(poseA[4]/2.0), poseB[1]+(poseB[4]/2.0))

    # compute the volume of intersection rectangle
    interArea = max(0, xmax - xmin) * max(0, ymax - ymin)
    # compute the volume of both the prediction and ground-truth object
    boxAArea = poseA[3] * poseA[4]
    boxBArea = poseB[3] * poseB[4]
    # compute the intersection over union by taking the intersection
    # volume and dividing it by the sum of prediction + ground-truth
    # volumes - the interesection volume
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

# non-max suppression
def non_max_suppression(label_dict_list, iou_threshold=0.1):
    if(len(label_dict_list) > 0):
        # delete lower-confidence objects with high iou
        i,j = 0,0
        while i < len(label_dict_list):
            while j < len(label_dict_list):
                if i != j:
                    if(iou(label_dict_list[i], label_dict_list[j]) > iou_threshold):
                        # if iou between two objects are higher than threshold
                        # then delete the object with lower confidence
                        if(label_dict_list[i]['conf'] >= label_dict_list[j]['conf']):
                            del label_dict_list[j]
                            j-=1
                        else:
                            del label_dict_list[i]
                            i-=1
                j+=1
            i+=1

    return label_dict_list

# convert 3d points to 2d
def pts_3d_to_2d(pts_3d, K, convert2uint=True):
    pts_2d = None
    if convert2uint == True:
        pts_2d = np.zeros((len(pts_3d), 2), dtype=np.uint16)
    else:
        pts_2d = np.zeros((len(pts_3d), 2), dtype=np.float32)
    for i in range(len(pts_3d)):
        if pts_3d[i][2] < 0.0:
            pts_3d[i][2] = 1e-6
        if(K.shape[1] == 4):
            pts_3d_ = np.append(np.transpose(pts_3d[i]), 1.0)
        else:
            pts_3d_ = np.transpose(pts_3d[i])

        pts_2d_ = np.matmul(K, pts_3d_)
        if convert2uint == True:
            pts_2d[i] = [np.uint16(max((pts_2d_[0]/pts_2d_[2]), 0)), np.uint16(max((pts_2d_[1]/pts_2d_[2]), 0))]
        else:
            pts_2d[i] = [max((pts_2d_[0]/pts_2d_[2]), 0.0), max((pts_2d_[1]/pts_2d_[2]), 0.0)]

    return np.array(pts_2d)

# convert labels from lidar frame to camera frame
def label_lidar2cam(label_dict, lidar2cam):
    label_dict_cam = []
    for label in label_dict:
        xyz1 = np.transpose([label['x'],label['y'],label['z'],1.0])
        xyz_cam = np.dot(lidar2cam, xyz1)
        label_cam = label.copy()
        label_cam['x'] = xyz_cam[0]
        label_cam['y'] = xyz_cam[1]
        label_cam['z'] = xyz_cam[2]
        label_dict_cam.append(label_cam)

    return label_dict_cam

# project lidar point-cloud to image plane
def project_pc2image(points, lidar2cam, K, resolution):
    '''
    Inputs:
        points: Nx3 vector containing lidar points
        lidar2cam: lidar-to-camera transformation matrix
        resolution: (x, y) resolution of camera frame
    Output:
        array of resolution (x, y) with lidar points projected on the image plane
    '''
    points_homogeneous = np.append(points, np.ones((points.shape[0],1)), axis=-1)
    points_cam = np.dot(lidar2cam, points_homogeneous.T).T
    points_img_homogeneous = np.dot(K, points_cam.T).T
    points_img = np.array([(p_/p_[-1])[:2] for p_ in points_img_homogeneous], dtype=np.int32)
    points_z = points_cam[:,2]
    # create mask
    points_mask = np.logical_and.reduce(((points_img[:,0] >= 0), (points_img[:,0] < resolution[0]), \
                                         (points_img[:,1] >= 0), (points_img[:,1] < resolution[1]), \
                                         (points_z > 0)))
    # filter out points
    points_img = points_img[points_mask]
    points_z = points_z[points_mask]
    # build 2d array for projected points
    projected_img = np.zeros((resolution[1], resolution[0]), dtype=np.float32)
    projected_img[(points_img[:,1],points_img[:,0])] = points_z

    # return image
    return projected_img

def colorize_depth_map(depth_map, min_depth=0, max_depth=100, cmap="magma", mask_zeros=False):

    # normalize 
    min_depth = depth_map.min() if min_depth is None else min_depth 
    max_depth = depth_map.max() if max_depth is None else max_depth  
    
    # apply mask
    if mask_zeros:
        mask = (depth_map == 0)

    # invert the scale for better colormap visualization 
    depth_map = max_depth - depth_map

    # scale between 0 to 1
    if min_depth != max_depth:
        depth_map = (depth_map - min_depth) / (max_depth - min_depth)
    else:
        depth_map = depth_map * 0  
    
    cmapper =  cm.get_cmap(cmap)
    depth_map = cmapper(depth_map, bytes=True) 
    img = depth_map[:,:,:3]

    if mask_zeros:
        img[mask] = (0, 0, 0)

    return img

def reproject_lr_disparity(img_left, img_right, depth_pred, f, baseline, camera):
    h, w = img_left.shape[-2], img_left.shape[-1] # could be a batch or single image

    resize = transforms.Compose([
                        transforms.Resize(size=(h, w))
                    ])
    # convert to tensor if depth is stored as numpy array
    depth_pred = resize(depth_pred)

    # huber norm
    huber_norm = torch.nn.SmoothL1Loss(reduction='none', beta=1.0)

    # compute depth
    disparity_1to2 = f * baseline / (depth_pred + 1e-6)
    
    # normalize disparity
    disparity_1to2 = disparity_1to2 / w

    img1 = img_left
    img2 = img_right
    # flip convention
    if camera == 'right':
        disparity_1to2 *= -1.0
        # flip images
        img1 = img_right
        img2 = img_left
    
    # warp left image to generate right image
    img2_warped = apply_disparity(img1, disparity_1to2)

    # get warped mask
    warping_mask_1to2 = torch.ones_like(img1)
    warping_mask_1to2 = apply_disparity(warping_mask_1to2, disparity_1to2)
    
    # compute left-to-right L1 loss
    # reproj_err_1to2 = warping_mask_1to2 * torch.abs(img2 - img2_warped)
    reproj_err_1to2 = warping_mask_1to2 * huber_norm(img2, img2_warped)

    # warp right image to generate left image
    img1_warped = apply_disparity(img2, -disparity_1to2)

    # get warped mask
    warping_mask_2to1 = torch.ones_like(img1)
    warping_mask_2to1 = apply_disparity(warping_mask_2to1, -disparity_1to2)

    # compute right-to-left L1 loss
    # reproj_err_2to1 = warping_mask_2to1 * torch.abs(img1 - img1_warped)
    reproj_err_2to1 = warping_mask_2to1 * huber_norm(img1, img1_warped)

    return depth_pred, disparity_1to2, img2_warped, warping_mask_1to2, reproj_err_1to2, img1_warped, warping_mask_2to1, reproj_err_2to1

def apply_disparity(img, disp):
    batch_size, _, height, width = img.shape

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(batch_size,
                height, 1).type_as(img)
    y_base = torch.linspace(0, 1, height).repeat(batch_size,
                width, 1).transpose(1, 2).type_as(img)

    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                           padding_mode='zeros')

    return output

def get_reprojection_vis(img_left, img_right, depth_pred, f, baseline):
    depth, disparity_l2r, img_right_warped, warping_mask_l2r, l2r_l1_err, img_left_warped, warping_mask_r2l, r2l_l1_err = \
        reproject_lr_disparity(img_left, img_right, depth_pred, f, baseline, camera='left')

    # left image
    img_l = img_left.detach().cpu().numpy()
    img_l = np.transpose(img_l[0], (1,2,0))
    img_l = np.array(img_l, dtype=np.uint8)

    # text on the visualization image
    x_center = int(img_l.shape[1] / 2)
    img_l = cv2.cvtColor(img_l, cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_l, (x_center-300,0), (x_center+300,60), color=(0,0,0), thickness=-1)
    cv2.putText(img_l, 'left image', 
                (x_center-100,40), 
                cv2.FONT_HERSHEY_COMPLEX, 
                1.0,
                color=(255,255,255),
                thickness=2,
                lineType=2)

    # right image
    img_r = img_right.detach().cpu().numpy()
    img_r = np.transpose(img_r[0], (1,2,0))
    img_r = np.array(img_r, dtype=np.uint8)

    # text on the visualization image
    img_r = cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR) 
    cv2.rectangle(img_r, (x_center-300,0), (x_center+300,60), color=(0,0,0), thickness=-1)
    cv2.putText(img_r, 'right image', 
        (x_center-100,40), 
        cv2.FONT_HERSHEY_COMPLEX, 
        1.0,
        color=(255,255,255),
        thickness=2,
        lineType=2)

    # depth map
    depth = depth.detach().cpu().numpy()
    depth = np.squeeze(depth[0], 0)
    depth_colorized = colorize_depth_map(depth)
    depth_colorized = np.array(depth_colorized, dtype=np.uint8)

    # text on the visualization image
    depth_colorized = cv2.cvtColor(depth_colorized, cv2.COLOR_RGB2BGR) 
    cv2.rectangle(depth_colorized, (x_center-300,0), (x_center+300,60), color=(0,0,0), thickness=-1)
    cv2.putText(depth_colorized, 'predicted depth', 
        (x_center-100,40), 
        cv2.FONT_HERSHEY_COMPLEX, 
        1.0,
        color=(255,255,255),
        thickness=2,
        lineType=2)

    # right-to-left warped image
    img_l_warped = img_left_warped.detach().cpu().numpy()
    img_l_warped = np.transpose(img_l_warped[0], (1,2,0))
    img_l_warped = np.array(img_l_warped, dtype=np.uint8)

    # # text on the visualization image
    img_l_warped = cv2.cvtColor(img_l_warped, cv2.COLOR_RGB2BGR) 
    cv2.rectangle(img_l_warped, (x_center-300,0), (x_center+300,60), color=(0,0,0), thickness=-1)
    cv2.putText(img_l_warped, 'right-to-left warped image', 
        (x_center-200,40), 
        cv2.FONT_HERSHEY_COMPLEX, 
        1.0,
        color=(255,255,255),
        thickness=2,
        lineType=2)

    # right-to-left reprojection error
    r2l_l1_err = r2l_l1_err.detach().cpu().numpy()
    r2l_l1_err = np.transpose(r2l_l1_err[0], (1,2,0))
    r2l_l1_err = np.array(r2l_l1_err, dtype=np.uint8)
    r2l_l1_err = cv2.cvtColor(r2l_l1_err, cv2.COLOR_RGB2GRAY) 

    # text on the visualization image
    r2l_l1_err = cv2.cvtColor(r2l_l1_err, cv2.COLOR_GRAY2BGR) 
    cv2.rectangle(r2l_l1_err, (x_center-300,0), (x_center+300,60), color=(0,0,0), thickness=-1)
    cv2.putText(r2l_l1_err, 'right-to-left reproj error', 
        (x_center-200,40), 
        cv2.FONT_HERSHEY_COMPLEX, 
        1.0,
        color=(255,255,255),
        thickness=2,
        lineType=2)

    # left-to-right warped image
    img_r_warped = img_right_warped.detach().cpu().numpy()
    img_r_warped = np.transpose(img_r_warped[0], (1,2,0))
    img_r_warped = np.array(img_r_warped, dtype=np.uint8)

    # text on the visualization image
    img_r_warped = cv2.cvtColor(img_r_warped, cv2.COLOR_RGB2BGR) 
    cv2.rectangle(img_r_warped, (x_center-300,0), (x_center+300,60), color=(0,0,0), thickness=-1)
    cv2.putText(img_r_warped, 'left-to-right warped image', 
        (x_center-200,40), 
        cv2.FONT_HERSHEY_COMPLEX, 
        1.0,
        color=(255,255,255),
        thickness=2,
        lineType=2)

    # right-to-left reprojection error
    l2r_l1_err = l2r_l1_err.detach().cpu().numpy()
    l2r_l1_err = np.transpose(l2r_l1_err[0], (1,2,0))
    l2r_l1_err = np.array(l2r_l1_err, dtype=np.uint8)
    l2r_l1_err = cv2.cvtColor(l2r_l1_err, cv2.COLOR_RGB2GRAY) 

    # text on the visualization image
    l2r_l1_err = cv2.cvtColor(l2r_l1_err, cv2.COLOR_GRAY2BGR) 
    cv2.rectangle(l2r_l1_err, (x_center-300,0), (x_center+300,60), color=(0,0,0), thickness=-1)
    cv2.putText(l2r_l1_err, 'left-to-right reproj error', 
        (x_center-200,40), 
        cv2.FONT_HERSHEY_COMPLEX, 
        1.0,
        color=(255,255,255),
        thickness=2,
        lineType=2)

    # visualization
    img_vis = cv2.vconcat([img_l, img_l_warped, r2l_l1_err, img_r, img_r_warped, l2r_l1_err, depth_colorized])
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB) 

    return img_vis