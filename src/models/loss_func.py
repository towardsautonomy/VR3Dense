import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from utils import label_map, pose_fields, reproject_lr_disparity, get_reprojection_vis, colorize_depth_map

# label vector length per object
obj_label_len = len(pose_fields) + len(label_map) # 9 for poses, rest for object classes

# cross entropy loss
def cross_entropy(y_true, y_pred, epsilon=1e-4):
    y_pred = torch.clamp(y_pred, min=0.0, max=1.0)
    ce = -torch.sum((y_true*torch.log(y_pred+epsilon) + (1.0-y_true)*torch.log(1.0-y_pred+epsilon)), dim=-1)
    return ce

# Mean IOU
class MeanIOU(nn.Module):
    def __init__(self):
        super(MeanIOU, self).__init__()

    def forward(self, y_true, y_pred):
        batch_size = y_true.size(0)
        y_pred = y_pred.view(batch_size, -1, obj_label_len).type(torch.FloatTensor)
        y_true = y_true.view(batch_size, -1, obj_label_len).type(torch.FloatTensor)
        
        # confidence
        true_conf = y_true[:,:,0]
        n_true_obj = torch.sum(true_conf)

        iou = box_iou(y_true, y_pred)
                
        # mean iou
        mean_iou = torch.sum(true_conf*iou) / (n_true_obj + 1e-6)
        return mean_iou

# Mean IOU loss
class IOULoss(nn.Module):
    def __init__(self, lambda_weight=1.0):
        super(IOULoss, self).__init__()
        self.lambda_weight = lambda_weight

    def forward(self, y_true, y_pred):
        batch_size = y_true.size(0)
        y_pred = y_pred.view(batch_size, -1, obj_label_len).type(torch.FloatTensor)
        y_true = y_true.view(batch_size, -1, obj_label_len).type(torch.FloatTensor)
        
        # confidence
        true_conf = y_true[:,:,0]
        n_true_obj = torch.sum(true_conf)

        iou = generalized_box_iou(y_true, y_pred)
        # iou = box_iou(y_true, y_pred)

        # iou loss
        iou_loss = torch.nn.MSELoss(reduction='none')(torch.ones_like(iou), iou)
        # iou_loss = -torch.log(iou + 1e-6)
        
        # mean giou
        iou_loss = torch.sum(true_conf*iou_loss) / (n_true_obj + 1e-6)

        return self.lambda_weight*iou_loss

class ConfLoss(nn.Module):
    def __init__(self, lambda_weight=1.0):
        super(ConfLoss, self).__init__()
        self.lambda_weight = lambda_weight

    def forward(self, y_true, y_pred):
        batch_size = y_true.size(0)
        y_pred = y_pred.view(batch_size, -1, obj_label_len).type(torch.FloatTensor)
        y_true = y_true.view(batch_size, -1, obj_label_len).type(torch.FloatTensor)
        n_possible_objs = y_true.size(1)

        # confidence
        true_conf = y_true[:,:,0]
        n_true_obj = torch.sum(true_conf)
        # true_ratio = n_true_obj / ((batch_size*n_possible_objs) - n_true_obj)

        # confidence loss | weight the losses accordingly
        conf_loss = torch.sum(true_conf*torch.nn.MSELoss(reduction='none')(y_true[:,:,0], y_pred[:,:,0])) / (n_true_obj + 1e-6) + \
                    torch.sum((1.0 - true_conf)*torch.nn.MSELoss(reduction='none')(y_true[:,:,0], y_pred[:,:,0])) / ((batch_size*n_possible_objs) - n_true_obj) 

        # return the loss
        return self.lambda_weight * conf_loss

class XLoss(nn.Module):
    def __init__(self, lambda_weight=1.0):
        super(XLoss, self).__init__()
        self.lambda_weight = lambda_weight

    def forward(self, y_true, y_pred):
        batch_size = y_true.size(0)
        y_pred = y_pred.view(batch_size, -1, obj_label_len).type(torch.FloatTensor)
        y_true = y_true.view(batch_size, -1, obj_label_len).type(torch.FloatTensor)

        # confidence
        true_conf = y_true[:,:,0]
        n_true_obj = torch.sum(true_conf)

        # x loss
        x_loss = torch.sum(true_conf*torch.nn.MSELoss(reduction='none')(y_true[:,:,1], y_pred[:,:,1]))  / (n_true_obj + 1e-6)

        # return the loss
        return self.lambda_weight * x_loss

class YLoss(nn.Module):
    def __init__(self, lambda_weight=1.0):
        super(YLoss, self).__init__()
        self.lambda_weight = lambda_weight

    def forward(self, y_true, y_pred):
        batch_size = y_true.size(0)
        y_pred = y_pred.view(batch_size, -1, obj_label_len).type(torch.FloatTensor)
        y_true = y_true.view(batch_size, -1, obj_label_len).type(torch.FloatTensor)

        # confidence
        true_conf = y_true[:,:,0]
        n_true_obj = torch.sum(true_conf)

        # y loss
        y_loss = torch.sum(true_conf*torch.nn.MSELoss(reduction='none')(y_true[:,:,2], y_pred[:,:,2]))  / (n_true_obj + 1e-6)

        # return the loss
        return self.lambda_weight * y_loss

class ZLoss(nn.Module):
    def __init__(self, lambda_weight=1.0):
        super(ZLoss, self).__init__()
        self.lambda_weight = lambda_weight

    def forward(self, y_true, y_pred):
        batch_size = y_true.size(0)
        y_pred = y_pred.view(batch_size, -1, obj_label_len).type(torch.FloatTensor)
        y_true = y_true.view(batch_size, -1, obj_label_len).type(torch.FloatTensor)

        # confidence
        true_conf = y_true[:,:,0]
        n_true_obj = torch.sum(true_conf)

        # z loss
        z_loss = torch.sum(true_conf*torch.nn.MSELoss(reduction='none')(y_true[:,:,3], y_pred[:,:,3]))  / (n_true_obj + 1e-6)

        # return the loss
        return self.lambda_weight * z_loss

class LLoss(nn.Module):
    def __init__(self, lambda_weight=1.0):
        super(LLoss, self).__init__()
        self.lambda_weight = lambda_weight

    def forward(self, y_true, y_pred):
        batch_size = y_true.size(0)
        y_pred = y_pred.view(batch_size, -1, obj_label_len).type(torch.FloatTensor)
        y_true = y_true.view(batch_size, -1, obj_label_len).type(torch.FloatTensor)

        # confidence
        true_conf = y_true[:,:,0]
        n_true_obj = torch.sum(true_conf)

        # l loss
        l_loss = torch.sum(true_conf*torch.nn.MSELoss(reduction='none')(y_true[:,:,4], y_pred[:,:,4]))  / (n_true_obj + 1e-6)

        # return the loss
        return self.lambda_weight * l_loss

class WLoss(nn.Module):
    def __init__(self, lambda_weight=1.0):
        super(WLoss, self).__init__()
        self.lambda_weight = lambda_weight

    def forward(self, y_true, y_pred):
        batch_size = y_true.size(0)
        y_pred = y_pred.view(batch_size, -1, obj_label_len).type(torch.FloatTensor)
        y_true = y_true.view(batch_size, -1, obj_label_len).type(torch.FloatTensor)

        # confidence
        true_conf = y_true[:,:,0]
        n_true_obj = torch.sum(true_conf)

        # w loss
        w_loss = torch.sum(true_conf*torch.nn.MSELoss(reduction='none')(y_true[:,:,5], y_pred[:,:,5]))  / (n_true_obj + 1e-6)

        # return the loss
        return self.lambda_weight * w_loss

class HLoss(nn.Module):
    def __init__(self, lambda_weight=1.0):
        super(HLoss, self).__init__()
        self.lambda_weight = lambda_weight

    def forward(self, y_true, y_pred):
        batch_size = y_true.size(0)
        y_pred = y_pred.view(batch_size, -1, obj_label_len).type(torch.FloatTensor)
        y_true = y_true.view(batch_size, -1, obj_label_len).type(torch.FloatTensor)

        # confidence
        true_conf = y_true[:,:,0]
        n_true_obj = torch.sum(true_conf)

        # H loss
        h_loss = torch.sum(true_conf*torch.nn.MSELoss(reduction='none')(y_true[:,:,6], y_pred[:,:,6]))  / (n_true_obj + 1e-6)

        # return the loss
        return self.lambda_weight * h_loss

class YawLoss(nn.Module):
    def __init__(self, lambda_weight=1.0):
        super(YawLoss, self).__init__()
        self.lambda_weight = lambda_weight

    def forward(self, y_true, y_pred):
        batch_size = y_true.size(0)
        y_pred = y_pred.view(batch_size, -1, obj_label_len).type(torch.FloatTensor)
        y_true = y_true.view(batch_size, -1, obj_label_len).type(torch.FloatTensor)

        # confidence
        true_conf = y_true[:,:,0]
        n_true_obj = torch.sum(true_conf)

        # Yaw loss
        cos_yaw_loss = torch.sum(torch.nn.MSELoss(reduction='none')(y_true[:,:,7], y_pred[:,:,7]))  / (n_true_obj + 1e-6)
        sin_yaw_loss = torch.sum(torch.nn.MSELoss(reduction='none')(y_true[:,:,8], y_pred[:,:,8]))  / (n_true_obj + 1e-6)
        yaw_loss = (cos_yaw_loss + sin_yaw_loss) / 2.0

        # return the loss
        return self.lambda_weight * yaw_loss

class ClassLoss(nn.Module):
    def __init__(self, lambda_weight=1.0):
        super(ClassLoss, self).__init__()
        self.lambda_weight = lambda_weight

    def forward(self, y_true, y_pred):
        batch_size = y_true.size(0)
        y_pred = y_pred.view(batch_size, -1, obj_label_len).type(torch.FloatTensor)
        y_true = y_true.view(batch_size, -1, obj_label_len).type(torch.FloatTensor)

        # confidence
        true_conf = y_true[:,:,0]
        n_true_obj = torch.sum(true_conf)

        # class loss
        class_loss = torch.sum(true_conf*cross_entropy(y_true[:,:,9:], y_pred[:,:,9:]))  / (n_true_obj + 1e-6)

        # return the loss
        return self.lambda_weight * class_loss

# bounding box area
def box_area(poses: torch.Tensor) -> torch.Tensor:
    """
    Computes the area of a set of object poses
    """
    return torch.exp(poses[:,:,pose_fields.index('l')]) * torch.exp(poses[:,:,pose_fields.index('w')])


# Implementation adapted from https://github.com/facebookresearch/detr/blob/master/util/box_ops.py
def box_iou(poses1: torch.Tensor, poses2: torch.Tensor) -> torch.Tensor:
    """
    Return intersection-over-union (Jaccard index) of object poses.
    """
    area1 = box_area(poses1)
    area2 = box_area(poses2)

    lt = torch.max(poses1[:,:,pose_fields.index('x'):pose_fields.index('y')+1]
                    -(torch.exp(poses1[:,:,pose_fields.index('l'):pose_fields.index('w')+1])/2.0), \
                   poses2[:,:,pose_fields.index('x'):pose_fields.index('y')+1]
                    -(torch.exp(poses2[:,:,pose_fields.index('l'):pose_fields.index('w')+1])/2.0))
    rb = torch.min(poses1[:,:,pose_fields.index('x'):pose_fields.index('y')+1]
                    +(torch.exp(poses1[:,:,pose_fields.index('l'):pose_fields.index('w')+1])/2.0), \
                   poses2[:,:,pose_fields.index('x'):pose_fields.index('y')+1]
                    +(torch.exp(poses2[:,:,pose_fields.index('l'):pose_fields.index('w')+1])/2.0))

    wh = (rb - lt).clamp(min=0) 
    # intersection
    inter = wh[:, :, 0] * wh[:, :, 1] 

    # union
    union = area1 + area2 - inter

    # intersection-over-union
    iou = inter / (union + 1e-6)

    return iou


# Implementation adapted from https://github.com/facebookresearch/detr/blob/master/util/box_ops.py
def generalized_box_iou(poses1: torch.Tensor, poses2: torch.Tensor) -> torch.Tensor:
    """
    Return generalized intersection-over-union (Jaccard index) of object poses.
    """

    area1 = box_area(poses1)
    area2 = box_area(poses2)

    lt = torch.max(poses1[:,:,pose_fields.index('x'):pose_fields.index('y')+1]
                    -(torch.exp(poses1[:,:,pose_fields.index('l'):pose_fields.index('w')+1])/2.0), \
                   poses2[:,:,pose_fields.index('x'):pose_fields.index('y')+1]
                    -(torch.exp(poses2[:,:,pose_fields.index('l'):pose_fields.index('w')+1])/2.0))
    rb = torch.min(poses1[:,:,pose_fields.index('x'):pose_fields.index('y')+1]
                    +(torch.exp(poses1[:,:,pose_fields.index('l'):pose_fields.index('w')+1])/2.0), \
                   poses2[:,:,pose_fields.index('x'):pose_fields.index('y')+1]
                    +(torch.exp(poses2[:,:,pose_fields.index('l'):pose_fields.index('w')+1])/2.0))

    wh = (rb - lt).clamp(min=0)  

    # intersection
    inter = wh[:, :, 0] * wh[:, :, 1] 

    # union
    union = area1 + area2 - inter

    # intersection-over-union
    iou = inter / (union + 1e-6)

    lti = torch.min(poses1[:,:,1:3]-(poses1[:,:,4:6]/2.0), \
                    poses2[:,:,1:3]-(poses2[:,:,4:6]/2.0))
    rbi = torch.max(poses1[:,:,1:3]+(poses1[:,:,4:6]/2.0), \
                    poses2[:,:,1:3]+(poses2[:,:,4:6]/2.0))

    whi = (rbi - lti).clamp(min=0) 
    areai = whi[:, :, 0] * whi[:, :, 1]

    return iou - (areai - union) / (areai + 1e-6)

class DepthL2Loss(nn.Module):
    def __init__(self, lambda_weight=0.2, decay_rate=0.01):
        super(DepthL2Loss, self).__init__()
        self.lambda_weight = lambda_weight
        self.decay_rate = decay_rate

    def forward(self, lidar_cam_projection, depth_pred, depth_mask, epoch):
        # decay the weight
        lambda_weight_decayed = self.lambda_weight - (epoch * self.lambda_weight * self.decay_rate)
        
        # create depth mask
        n_true_pts = torch.sum(depth_mask)

        depth_l2_loss = lambda_weight_decayed * (torch.sum(depth_mask * (lidar_cam_projection - depth_pred) ** 2) / n_true_pts)

        # return the loss
        return depth_l2_loss

class DepthSmoothnessLoss(nn.Module):
    def __init__(self, lambda_weight=1.0, alpha=0.5):
        super(DepthSmoothnessLoss, self).__init__()
        self.lambda_weight = lambda_weight
        self.beta_edge = alpha
        self.edge_preserverance_loss = DepthEdgePreserveranceLoss()
        # grayscale transformation
        self.to_grayscale = transforms.Compose([
                            transforms.Grayscale(num_output_channels=1)
                        ])

    def forward(self, img, depth_pred):
        img_gray = self.to_grayscale(img)

        img_x_grad = img_gray[:,:,:,:-1] - img_gray[:,:,:,1:]
        img_y_grad = img_gray[:,:,:-1,:] - img_gray[:,:,1:,:]

        depth_x_grad = depth_pred[:,:,:,:-1] - depth_pred[:,:,:,1:]
        depth_y_grad = depth_pred[:,:,:-1,:] - depth_pred[:,:,1:,:]

        # penalize sharper edges if a corresponding edge unavailable in input image
        edge_aware_smooth_loss_ = torch.mean(torch.abs(depth_x_grad) * torch.exp(-torch.abs(img_x_grad))) + torch.mean(torch.abs(depth_y_grad) * torch.exp(-torch.abs(img_y_grad)))
        
        # preserve edges
        edge_preserverance_loss_ = self.edge_preserverance_loss(img, depth_pred)

        return self.lambda_weight * ((self.beta_edge * edge_preserverance_loss_) + ((1. - self.beta_edge) * edge_aware_smooth_loss_))

class DepthEdgePreserveranceLoss(nn.Module):
    def __init__(self, lambda_weight=1.0, device='cuda'):
        super(DepthEdgePreserveranceLoss, self).__init__()
        self.lambda_weight = lambda_weight
        self.alpha = torch.Tensor([1., 1.])
        # self.W = nn.Parameter(torch.Tensor([0.99769384, 0.99669755]))
        # self.b = nn.Parameter(torch.Tensor([ 0.00058065, -0.00093009]))
        
        self.W = torch.Tensor([0.99769384, 0.99669755])
        self.b = torch.Tensor([0.00058065, -0.00093009])
        
        # grayscale transformation
        self.to_grayscale = transforms.Compose([
                            transforms.Grayscale(num_output_channels=1)
                        ])

    def forward(self, img, depth_pred):
        img_gray = self.to_grayscale(img)

        img_x_grad = img_gray[:,:,:,:-1] - img_gray[:,:,:,1:]
        img_y_grad = img_gray[:,:,:-1,:] - img_gray[:,:,1:,:]

        depth_x_grad = depth_pred[:,:,:,:-1] - depth_pred[:,:,:,1:]
        depth_y_grad = depth_pred[:,:,:-1,:] - depth_pred[:,:,1:,:]

        # learn alpha
        self.alpha[0] = F.tanh(torch.mean(img_x_grad * self.W[0] + self.b[0]))
        self.alpha[1] = F.tanh(torch.mean(img_y_grad * self.W[1] + self.b[1]))
        
        # enforce similar edge between image and depth
        edge_preserverance_loss = (torch.mean(torch.exp(torch.abs(depth_x_grad - self.alpha[0]*img_x_grad))) + torch.mean(torch.exp(torch.abs(depth_y_grad - self.alpha[1]*img_y_grad))) / 2.0) - 1.0

        return self.lambda_weight * edge_preserverance_loss

class DepthUnsupervisedLoss(nn.Module):
    '''
    Inspired by Monodepth: https://arxiv.org/pdf/1609.03677.pdf
    '''
    def __init__(self, lambda_weight=1.0):
        super(DepthUnsupervisedLoss, self).__init__()
        self.lambda_weight = lambda_weight
        self.alpha = 0.85
        # huber norm
        self.huber_norm = torch.nn.SmoothL1Loss(reduction='mean', beta=1.0)

    def forward(self, img_left, img_right, depth_pred_left, depth_pred_right, f, baseline):
        depth_l, disparity_l2r, img_right_warped, warping_mask_l2r, l2r_repr_err, img_left_warped, warping_mask_r2l, r2l_repr_err = \
            reproject_lr_disparity(img_left, img_right, depth_pred_left, f, baseline, camera='left')

        depth_r, disparity_r2l, img_left_warped_2, warping_mask_r2l_2, r2l_repr_err_2, img_right_warped_2, warping_mask_l2r_2, l2r_repr_err_2 = \
            reproject_lr_disparity(img_left, img_right, depth_pred_right, f, baseline, camera='right')

        # compute reprojection loss
        l2r_repr_loss = torch.mean(l2r_repr_err)
        r2l_repr_loss = torch.mean(r2l_repr_err)
        l2r_repr_loss_2 = torch.mean(l2r_repr_err_2)
        r2l_repr_loss_2 = torch.mean(r2l_repr_err_2)
        recon_repr_loss = (l2r_repr_loss + r2l_repr_loss + l2r_repr_loss_2 + r2l_repr_loss_2) / 4.

        # compute SSIM
        ssim_l2r = ssim(img_left, img_left_warped)
        ssim_r2l = ssim(img_right, img_right_warped)
        ssim_l2r_2 = ssim(img_left, img_left_warped_2)
        ssim_r2l_2 = ssim(img_right, img_right_warped_2)
        ssim_loss = (ssim_l2r + ssim_r2l + ssim_l2r_2 + ssim_r2l_2) / 4.

        # appearance matching loss
        appearance_match_loss = (self.alpha * (1. - ssim_loss) / 2.0) + ((1. - self.alpha) * recon_repr_loss)

        # left-right disparity consistency loss
        lr_consistency_loss = self.huber_norm(disparity_l2r, -disparity_r2l)
        
        return self.lambda_weight * (appearance_match_loss + lr_consistency_loss)

'''
Implementation of SSIM referred from: https://github.com/ialhashim/DenseDepth/blob/master/PyTorch/loss.py
'''
def ssim(img1, img2, window_size=3, window=None, size_average=True, full=False):

    pad = window_size // 2
    
    try:
        _, channels, height, width = img1.size()
    except:
        channels, height, width = img1.size()

    # if window is not provided, init one
    if window is None: 
        real_size = min(window_size, height, width) # window should be atleast 11x11 
        window = create_window(real_size, channel=channels).to(img1.device)
    
    # calculating the mu parameter (locally) for both images using a gaussian filter 
    # calculates the luminosity params
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2 
    mu12 = mu1 * mu2

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 =  F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

    # Some constants for stability 
    C1 = (0.01 ) ** 2  
    C2 = (0.03 ) ** 2 

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    contrast_metric = torch.mean(contrast_metric)

    numerator1 = 2 * mu12 + C1  
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1 
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

    if size_average:
        ret = ssim_score.mean() 
    else: 
        ret = ssim_score.mean(1).mean(1).mean(1)
    
    if full:
        return ret, contrast_metric
    
    return ret
    
def gaussian(window_size, sigma):
    gauss =  torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)

    window = torch.Tensor(_2D_window.expand(channel, 1, window_size, window_size).contiguous())

    return window