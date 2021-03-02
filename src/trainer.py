import os
import sys

# add path to sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.abspath('src'))
sys.path.append(BASE_DIR)
sys.path.append(SRC_DIR)

import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import datetime
import time
import random
import csv
from src.utils import *
from models.loss_func import *

# get device info
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Trainer class
class Trainer(object):

    def __init__(self, dataroot, model, dataset, dense_depth,
                       n_xgrids, n_ygrids, epochs, batch_size, learning_rate, exp_str,
                       xmin, xmax, ymin, ymax, zmin, zmax,
                       vol_size_x, vol_size_y, vol_size_z, img_size_x, img_size_y,
                       mode='train', modeldir='./models', logdir='./logs', plotdir='./plots',
                       model_save_steps=10, early_stop_steps=10, test_split=0.1,):
        """
        Parameter initialization

        Inputs:
            dataroot: folder that stores images for each modality
            model: the created model
            dataset: the dataset to use
            n_xgrids: number of grids in the x direction
            n_ygrids: number of grids in the y direction
            epochs: number of epochs to train the model
            batch_size: batch size for generating data during training
            learning_rate: learning rate for the optimizer
            exp_str: experiment string
            modeldir: directory path to save checkpoints
            logdir: directory path to save logs
            plotdir: directory path to save plots
            model_save_steps: number of steps to save the model checkpoint
            early_stop_steps: number of steps to wait before terminating model training if there is no improvement.
        """
        self.dataroot = dataroot
        self.model = model
        if torch.cuda.is_available():
            self.model.to(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_xgrids = n_xgrids
        self.n_ygrids = n_ygrids
        self.modeldir = modeldir
        self.logdir = logdir
        self.plotdir = plotdir
        self.model_save_steps = model_save_steps
        self.early_stop_steps = early_stop_steps
        self.exp_str = exp_str
        self.dense_depth = dense_depth
        # axes limits
        self.xlim = [xmin, xmax]
        self.ylim = [ymin, ymax]
        self.zlim = [zmin, zmax]
        # volume size
        self.vol_size = (vol_size_x, vol_size_y, vol_size_z)
        # image size
        self.img_size = (img_size_x, img_size_y)
        # mode
        if mode == 'train':
            # dataset
            self.dataset = dataset(dataroot, n_xgrids=n_xgrids, n_ygrids=n_ygrids, \
                                    xlim=self.xlim, ylim=self.ylim, zlim=self.zlim,
                                    vol_size=self.vol_size, img_size=self.img_size)
            # train test split
            self.test_split = test_split
            self.train_test_lengths = [int(len(self.dataset) - int(len(self.dataset)*self.test_split)), int(len(self.dataset)*self.test_split)]

            self.train_dataset, self.test_dataset = random_split(self.dataset, self.train_test_lengths)
            # dataloaders
            if(len(self.train_dataset) > 0):
                self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=4, collate_fn=self.collate_fn)
            if(len(self.test_dataset) > 0):
                self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=4, collate_fn=self.collate_fn)
                                
            # optimizer
            self.optimizer  = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate)
            # loss functions
            self.conf_loss = ConfLoss(lambda_weight=1.0)
            self.x_loss = XLoss(lambda_weight=1.0)
            self.y_loss = YLoss(lambda_weight=1.0)
            self.z_loss = ZLoss(lambda_weight=1.0)
            self.l_loss = LLoss(lambda_weight=1.0)
            self.w_loss = WLoss(lambda_weight=1.0)
            self.h_loss = HLoss(lambda_weight=1.0)
            self.yaw_loss = YawLoss(lambda_weight=1.0)
            self.iou_loss = IOULoss(lambda_weight=0.1)
            self.class_loss = ClassLoss(lambda_weight=0.2)
            self.depth_unsupervised_loss = DepthUnsupervisedLoss(lambda_weight=1.0)
            self.depth_l2_loss = DepthL2Loss(lambda_weight=0.1)
            self.depth_smoothness_loss = DepthSmoothnessLoss(lambda_weight=0.1)
            # Mean IOU
            self.mean_iou = MeanIOU()

    def collate_fn(self, batch):
        return zip(*[(b['cloud_voxelized'], b['label_vector'], b['left_image_resized'], b['right_image_resized'], b['lidar_cam_projection'], b['left_image'], b['right_image']) for b in batch])

    def train(self):
        """
        Load corresponding data and start training
        """
        # csv writer to log losses
        csv_dir = os.path.join(self.logdir, 'csv')
        csv_logfile = os.path.join(csv_dir, self.exp_str + '.csv')
        # tensorboard_dir = os.path.join(self.logdir, self.exp_str, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        # get experiment id
        exp_params, exp_id = self.exp_str.split('.exp_id_', 1)
        tensorboard_dir = os.path.join(self.logdir, exp_params, exp_id)
        model_exp_dir = os.path.join(self.modeldir, self.exp_str)
        # make directories
        os.system('mkdir -p {}'.format(csv_dir))
        os.system('mkdir -p {}'.format(model_exp_dir))
        
        print('==========================================================')
        print('Logging to csv -> {}'.format(csv_logfile))
        print('Logging to tensorboard -> {}'.format(tensorboard_dir))
        print('Training with {} training and {} testing samples.'.format(len(self.train_dataset), len(self.test_dataset)))
        print('==========================================================')

        # tensorboard writer
        tb_summary_writer = SummaryWriter(log_dir=tensorboard_dir)

        # model    
        self.model = self.model.to(device)

        # write graph to tensorboard
        dummy_x_lidar = torch.randn(1,1,self.vol_size[2],self.vol_size[0],self.vol_size[1]).to(device)
        dummy_x_camera = torch.randn(1,3,self.img_size[0],self.img_size[1]).to(device)
        tb_summary_writer.add_graph(self.model, (dummy_x_lidar, dummy_x_camera))
        
        with open(csv_logfile, 'w', newline='') as csvfile:
            fieldnames = ['epoch',                       \
                          'train_conf_loss',             \
                          'train_x_loss',                \
                          'train_y_loss',                \
                          'train_z_loss',                \
                          'train_l_loss',                \
                          'train_w_loss',                \
                          'train_h_loss',                \
                          'train_yaw_loss',              \
                          'train_iou_loss',              \
                          'train_class_loss',            \
                          'train_appearance_match_loss', \
                          'train_depth_l2_loss',         \
                          'train_depth_smooth_loss',     \
                          'train_total_loss',            \
                          'valid_conf_loss',             \
                          'valid_x_loss',                \
                          'valid_y_loss',                \
                          'valid_z_loss',                \
                          'valid_l_loss',                \
                          'valid_w_loss',                \
                          'valid_h_loss',                \
                          'valid_yaw_loss',              \
                          'valid_iou_loss',              \
                          'valid_class_loss',            \
                          'valid_appearance_match_loss', \
                          'valid_depth_l2_loss',         \
                          'valid_depth_smooth_loss',     \
                          'valid_total_loss'             ]
            # write header
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            csv_writer.writeheader()

            # minimum loss
            min_train_mean_loss = 1e6
            n_consecutive_epochs_no_improvement = 0
            for epoch in range(self.epochs):
                # train for one epoch
                start_time = time.time()
                # set model in train mode 
                self.model.train()
                train_losses = []
                # iterate over mini batches
                for i_batch, sample_batched in enumerate(self.train_dataloader):
                    # get data and labels
                    x_lidar_batch, y_pose_batch, x_left_batch, x_right_batch, y_lidar_cam_projection_batch, left_img_batch, right_img_batch = sample_batched
                    x_lidar_batch = np.expand_dims(x_lidar_batch, 1).astype(np.float32)
                    x_lidar_batch = torch.from_numpy(x_lidar_batch).to(device)
                    y_pose_batch = torch.Tensor(y_pose_batch).to(device)
                    x_left_batch = torch.from_numpy(np.asarray(x_left_batch, dtype=np.float32)).to(device)
                    x_left_batch = normalize_img(x_left_batch)
                    x_right_batch = torch.from_numpy(np.asarray(x_right_batch, dtype=np.float32)).to(device)
                    x_right_batch = normalize_img(x_right_batch)
                    y_lidar_cam_projection_batch = torch.from_numpy(np.asarray(y_lidar_cam_projection_batch, dtype=np.float32)).to(device)

                    # set optimizer grads to zero
                    self.optimizer.zero_grad()
            
                    # forward pass
                    pred_tuple = self.model(x_lidar_batch, x_left_batch)
                    if self.dense_depth:
                        pose_pred, depth_pred_left = pred_tuple
                        _, depth_pred_right = self.model(x_lidar_batch, x_right_batch)
                        _.detach()
                    else:
                        pose_pred = pred_tuple

                    # compute loss
                    conf_loss = self.conf_loss(y_pose_batch, pose_pred)
                    x_loss = self.x_loss(y_pose_batch, pose_pred)
                    y_loss = self.y_loss(y_pose_batch, pose_pred)
                    z_loss = self.z_loss(y_pose_batch, pose_pred)
                    l_loss = self.l_loss(y_pose_batch, pose_pred)
                    w_loss = self.w_loss(y_pose_batch, pose_pred)
                    h_loss = self.h_loss(y_pose_batch, pose_pred)
                    yaw_loss = self.yaw_loss(y_pose_batch, pose_pred)
                    iou_loss = self.iou_loss(y_pose_batch, pose_pred)
                    class_loss = self.class_loss(y_pose_batch, pose_pred)
                    depth_l2_loss = torch.Tensor([0.])
                    depth_smooth_loss = torch.Tensor([0.])
                    depth_unsupervised_loss = torch.Tensor([0.])
                    if self.dense_depth:
                        # reprojection loss
                        left_img_batch = torch.from_numpy(np.asarray(left_img_batch, dtype=np.float32)).to(device)
                        left_img_batch = normalize_img(left_img_batch)
                        right_img_batch = torch.from_numpy(np.asarray(right_img_batch, dtype=np.float32)).to(device)
                        right_img_batch = normalize_img(right_img_batch)

                        depth_unsupervised_loss = self.depth_unsupervised_loss(left_img_batch, right_img_batch, denormalize_depth(depth_pred_left, self.xlim[1]), denormalize_depth(depth_pred_right, self.xlim[1]), self.dataset.fx, self.dataset.baseline)
                        # l2 loss
                        depth_mask = (y_lidar_cam_projection_batch > 0)
                        depth_l2_loss = self.depth_l2_loss(normalize_depth(y_lidar_cam_projection_batch, self.xlim[1]), depth_pred_left, depth_mask)
                        # edge-aware-smoothing loss
                        depth_smooth_loss = self.depth_smoothness_loss(x_left_batch, depth_pred_left)
                    depth_loss = depth_unsupervised_loss + depth_l2_loss + depth_smooth_loss

                    obj_loss = conf_loss + x_loss + y_loss + z_loss + l_loss + w_loss + h_loss + yaw_loss + iou_loss + class_loss
                    total_loss = obj_loss + depth_loss

                    # compute mean iou
                    train_mean_iou = self.mean_iou(y_pose_batch, pose_pred).item()
                    # perform back-propagation
                    total_loss.backward()
                    # one optimization step
                    self.optimizer.step()
                    # add to dictionary for logs
                    train_loss = {  'conf_loss':               conf_loss.item(), 
                                    'x_loss':                  x_loss.item(), 
                                    'y_loss':                  y_loss.item(),
                                    'z_loss':                  z_loss.item(),
                                    'l_loss':                  l_loss.item(),
                                    'w_loss':                  w_loss.item(),
                                    'h_loss':                  h_loss.item(),
                                    'yaw_loss':                yaw_loss.item(),
                                    'iou_loss':                iou_loss.item(),
                                    'class_loss':              class_loss.item(),
                                    'depth_unsupervised_loss': depth_unsupervised_loss.item(),
                                    'depth_l2_loss':           depth_l2_loss.item(),
                                    'depth_smooth_loss':       depth_smooth_loss.item(),
                                    'total_loss':              total_loss.item()}
                    train_losses.append(total_loss.item())
                    
                    # log to tensorboard
                    for key_ in train_loss.keys():
                        tb_summary_writer.add_scalar('train/{}'.format(key_), train_loss[key_], global_step=epoch)
                    tb_summary_writer.add_scalar('train/mean_iou', train_mean_iou, global_step=epoch)
                    
                    # log time
                    end_time = time.time()
                    time_taken = end_time - start_time
                    time_taken_formatted = time.strftime("%H:%M:%S", time.gmtime(time_taken))
                    time_per_batch = time_taken / (i_batch + 1)
                    time_remaining = max(0.0, ((len(self.train_dataloader) * time_per_batch) - time_taken))
                    time_remaining_formatted = time.strftime("%H:%M:%S", time.gmtime(time_remaining))
                    # print stats
                    print('\rEpoch: [{:4d}] | Step: [{:4d}] | Training Loss [conf: {:.4f}, x: {:.4f}, y: {:.4f}, z: {:.4f}, l: {:.4f}, w: {:.4f}, h: {:.4f}, yaw: {:.4f}, iou: {:.4f}, class: {:.4f}, depth_unsup: {:.4f}, depth_l2: {:.4f}, depth_smooth: {:.4f}, total: {:.4f}] | Mean IOU: {:.4f} | time taken: {} | time remaining: {}       '.format(
                                epoch, i_batch, train_loss['conf_loss'], train_loss['x_loss'], train_loss['y_loss'], train_loss['z_loss'], train_loss['l_loss'], train_loss['w_loss'], 
                                train_loss['h_loss'], train_loss['yaw_loss'], train_loss['iou_loss'], train_loss['class_loss'], train_loss['depth_unsupervised_loss'], train_loss['depth_l2_loss'], 
                                train_loss['depth_smooth_loss'], train_loss['total_loss'], train_mean_iou, time_taken_formatted, time_remaining_formatted), end='')
                
                    # # uncomment if you want to limit number of steps per epoch
                    # if i_batch == 5:
                    #     break

                # compute mean loss
                train_mean_loss = np.mean(train_losses[-10:])

                # perform validation
                # set model in eval mode 
                self.model.eval()
                val_conf_losses,val_x_losses,val_y_losses,val_z_losses,\
                    val_l_losses,val_w_losses,val_h_losses,val_yaw_losses,\
                    val_iou_losses, val_class_losses,val_depth_unsup_losses,\
                        val_depth_l2_losses,val_depth_smooth_losses,val_total_losses = \
                        [],[],[],[],[],[],[],[],[],[],[],[],[],[]

                val_mean_iou_list = []
                # iterate over mini batches
                for i_batch, sample_batched in enumerate(self.test_dataloader):
                    # get data and labels
                    x_lidar_batch, y_pose_batch, x_left_batch, x_right_batch, y_lidar_cam_projection_batch, left_img_batch, right_img_batch = sample_batched
                    x_lidar_batch = np.expand_dims(x_lidar_batch, 1).astype(np.float32)
                    x_lidar_batch = torch.from_numpy(x_lidar_batch).to(device)
                    y_pose_batch = torch.Tensor(y_pose_batch).to(device)
                    x_left_batch = torch.from_numpy(np.asarray(x_left_batch, dtype=np.float32)).to(device)
                    x_left_batch = normalize_img(x_left_batch)
                    x_right_batch = torch.from_numpy(np.asarray(x_right_batch, dtype=np.float32)).to(device)
                    x_right_batch = normalize_img(x_right_batch)
                    y_lidar_cam_projection_batch = torch.from_numpy(np.asarray(y_lidar_cam_projection_batch, dtype=np.float32)).to(device)

                    # forward pass
                    with torch.no_grad():
                        pred_tuple = self.model(x_lidar_batch, x_left_batch)
                        if self.dense_depth:
                            pose_pred, depth_pred_left = pred_tuple
                            _, depth_pred_right = self.model(x_lidar_batch, x_right_batch)
                        else:
                            pose_pred = pred_tuple

                    # compute loss
                    conf_loss = self.conf_loss(y_pose_batch, pose_pred)
                    x_loss = self.x_loss(y_pose_batch, pose_pred)
                    y_loss = self.y_loss(y_pose_batch, pose_pred)
                    z_loss = self.z_loss(y_pose_batch, pose_pred)
                    l_loss = self.l_loss(y_pose_batch, pose_pred)
                    w_loss = self.w_loss(y_pose_batch, pose_pred)
                    h_loss = self.h_loss(y_pose_batch, pose_pred)
                    yaw_loss = self.yaw_loss(y_pose_batch, pose_pred)
                    iou_loss = self.iou_loss(y_pose_batch, pose_pred)
                    class_loss = self.class_loss(y_pose_batch, pose_pred)
                    depth_l2_loss = torch.Tensor([0.])
                    depth_smooth_loss = torch.Tensor([0.])
                    depth_unsupervised_loss = torch.Tensor([0.])
                    if self.dense_depth:
                        # reprojection loss
                        left_img_batch = torch.from_numpy(np.asarray(left_img_batch, dtype=np.float32)).to(device)
                        left_img_batch = normalize_img(left_img_batch)
                        right_img_batch = torch.from_numpy(np.asarray(right_img_batch, dtype=np.float32)).to(device)
                        right_img_batch = normalize_img(right_img_batch)

                        depth_unsupervised_loss = self.depth_unsupervised_loss(left_img_batch, right_img_batch, denormalize_depth(depth_pred_left, self.xlim[1]), denormalize_depth(depth_pred_right, self.xlim[1]), self.dataset.fx, self.dataset.baseline)
                        # l2 loss
                        depth_mask = (y_lidar_cam_projection_batch > 0)
                        depth_l2_loss = self.depth_l2_loss(normalize_depth(y_lidar_cam_projection_batch, self.xlim[1]), depth_pred_left, depth_mask)
                        # edge-aware-smoothing loss
                        depth_smooth_loss = self.depth_smoothness_loss(x_left_batch, depth_pred_left)
                    depth_loss = depth_unsupervised_loss + depth_l2_loss + depth_smooth_loss

                    obj_loss = conf_loss + x_loss + y_loss + z_loss + l_loss + w_loss + h_loss + yaw_loss + iou_loss + class_loss
                    total_loss = obj_loss + depth_loss
                    # compute mean iou
                    val_mean_iou = self.mean_iou(y_pose_batch, pose_pred)

                    # add to dictionary for logs
                    val_conf_losses.append(conf_loss.item())
                    val_x_losses.append(x_loss.item())
                    val_y_losses.append(y_loss.item())
                    val_z_losses.append(z_loss.item())
                    val_l_losses.append(l_loss.item())
                    val_w_losses.append(w_loss.item())
                    val_h_losses.append(h_loss.item())
                    val_yaw_losses.append(yaw_loss.item())
                    val_iou_losses.append(iou_loss.item())
                    val_class_losses.append(class_loss.item())
                    val_depth_unsup_losses.append(depth_unsupervised_loss.item())
                    val_depth_l2_losses.append(depth_l2_loss.item())
                    val_depth_smooth_losses.append(depth_smooth_loss.item())
                    val_total_losses.append(total_loss.item())
                    val_mean_iou_list.append(val_mean_iou.item())

                    # perform validation only on 2 batches
                    if i_batch == 1:
                        # add visualizations to tensorboard
                        x_lidar_batch, y_pose_batch, x_left_batch, x_right_batch, y_lidar_cam_projection_batch, left_img_batch, right_img_batch = next(iter(self.test_dataloader))
                        sample_test_indices = random.sample(range(len(self.test_dataset)), 4)
                        for k, sample_idx in enumerate(sample_test_indices):
                            sample = self.test_dataset[sample_idx]
                            ## get true labels visualization
                            pc_bbox_img_true = draw_point_cloud_w_bbox(sample['cloud'], sample['label_dict'], \
                                                                        xlim=self.xlim, ylim=self.ylim, zlim=self.zlim)
                            tb_summary_writer.add_image('ground-truth-object/{}'.format(k), pc_bbox_img_true, global_step=epoch, dataformats='HWC')

                            # get predicted poses and classes
                            pred_tuple = self.predict(sample['cloud'], sample['left_image'])
                            if self.dense_depth:
                                lidar_cam_projection = np.squeeze(sample['lidar_cam_projection'], 0)
                                label_dict, dense_depth = pred_tuple

                                # colorize depth maps for visualization
                                depth_true_colorized = colorize_depth_map(lidar_cam_projection, 0., self.dataset.xlim[1])
                                depth_pred_colorized = colorize_depth_map(dense_depth, 0., self.dataset.xlim[1])

                                tb_summary_writer.add_image('left-camera-image/{}'.format(k), np.transpose(sample['left_image_resized'], (1,2,0)), global_step=epoch, dataformats='HWC')
                                tb_summary_writer.add_image('ground-truth-depth/{}'.format(k), depth_true_colorized, global_step=epoch, dataformats='HWC')
                                tb_summary_writer.add_image('prediction-depth/{}'.format(k), depth_pred_colorized, global_step=epoch, dataformats='HWC')

                                # reprojection
                                dense_depth = dense_depth[np.newaxis, np.newaxis, ...]
                                dense_depth = torch.from_numpy(np.asarray(dense_depth, dtype=np.float32)).to(device)
                                left_image_torch = sample['left_image'][np.newaxis, ...]
                                left_image_torch = torch.from_numpy(np.asarray(left_image_torch, dtype=np.float32)).to(device)
                                right_image_torch = sample['right_image'][np.newaxis, ...]
                                right_image_torch = torch.from_numpy(np.asarray(right_image_torch, dtype=np.float32)).to(device)
                                vis_img = get_reprojection_vis(left_image_torch, right_image_torch, dense_depth, self.dataset.fx, self.dataset.baseline)
                                tb_summary_writer.add_image('left-right-reprojection/{}'.format(k), vis_img, global_step=epoch, dataformats='HWC')
                            else:
                                label_dict = pred_tuple

                            ## get true labels visualization
                            pc_bbox_img_pred = draw_point_cloud_w_bbox(sample['cloud'], label_dict, \
                                                                        xlim=self.xlim, ylim=self.ylim, zlim=self.zlim)
                            tb_summary_writer.add_image('prediction-object/{}'.format(k), pc_bbox_img_pred, global_step=epoch, dataformats='HWC')

                        # break loop
                        break
                
                # add to dictionary for logs
                val_loss = {  'conf_loss':                   np.mean(val_conf_losses), 
                              'x_loss':                      np.mean(val_x_losses),
                              'y_loss':                      np.mean(val_y_losses),
                              'z_loss':                      np.mean(val_z_losses),
                              'l_loss':                      np.mean(val_l_losses),
                              'w_loss':                      np.mean(val_w_losses),
                              'h_loss':                      np.mean(val_h_losses),
                              'yaw_loss':                    np.mean(val_yaw_losses),
                              'iou_loss':                    np.mean(val_iou_losses),
                              'class_loss':                  np.mean(val_class_losses),
                              'depth_unsupervised_loss': np.mean(val_depth_unsup_losses),
                              'depth_l2_loss':               np.mean(val_depth_l2_losses),
                              'depth_smooth_loss':           np.mean(val_depth_smooth_losses),
                              'total_loss':                  np.mean(val_total_losses)}
                    
                # log to tensorboard
                for key_ in val_loss.keys():
                    tb_summary_writer.add_scalar('test/{}'.format(key_), val_loss[key_], global_step=epoch)
                tb_summary_writer.add_scalar('test/mean_iou', np.mean(val_mean_iou_list), global_step=epoch)
                    
                # print epoch summary
                print('\r                                               \
                                                                        \
                                                                        \
                                                                        \
                                                                        ', end='')
                print('\rEpoch: [{:4d}]   | \
                        \nTraining Loss   | [conf: {:.4f}, x: {:.4f}, y: {:.4f}, z: {:.4f}, l: {:.4f}, w: {:.4f}, h: {:.4f}, yaw: {:.4f}, iou: {:.4f}, class: {:.4f}, depth_unsup: {:.4f}, depth_l2: {:.4f}, depth_smooth: {:.4f}, total: {:.4f}] | Mean IOU: {:.4f} | time taken: {}'.format(
                            epoch, train_loss['conf_loss'], train_loss['x_loss'], train_loss['y_loss'], train_loss['z_loss'], train_loss['l_loss'], train_loss['w_loss'], 
                            train_loss['h_loss'], train_loss['yaw_loss'], train_loss['iou_loss'], train_loss['class_loss'], train_loss['depth_unsupervised_loss'], train_loss['depth_l2_loss'], train_loss['depth_smooth_loss'], 
                            train_loss['total_loss'], train_mean_iou, time_taken_formatted))
                print('Validation Loss | [conf: {:.4f}, x: {:.4f}, y: {:.4f}, z: {:.4f}, l: {:.4f}, w: {:.4f}, h: {:.4f}, yaw: {:.4f}, iou: {:.4f}, class: {:.4f}, depth_unsup: {:.4f}, depth_l2: {:.4f}, depth_smooth: {:.4f}, total: {:.4f}] | Mean IOU: {:.4f}'.format(
                            val_loss['conf_loss'], val_loss['x_loss'], val_loss['y_loss'], val_loss['z_loss'], val_loss['l_loss'], val_loss['w_loss'], 
                            val_loss['h_loss'], val_loss['yaw_loss'], val_loss['iou_loss'], val_loss['class_loss'], val_loss['depth_unsupervised_loss'], val_loss['depth_l2_loss'], val_loss['depth_smooth_loss'], 
                            val_loss['total_loss'], np.mean(val_mean_iou_list)))
                
                # write to logfile
                csv_writer.writerow({'epoch'                       : epoch,                            \
                                     'train_conf_loss'             : train_loss['conf_loss'],          \
                                     'train_x_loss'                : train_loss['x_loss'],             \
                                     'train_y_loss'                : train_loss['x_loss'],             \
                                     'train_z_loss'                : train_loss['z_loss'],             \
                                     'train_l_loss'                : train_loss['l_loss'],             \
                                     'train_w_loss'                : train_loss['w_loss'],             \
                                     'train_h_loss'                : train_loss['h_loss'],             \
                                     'train_yaw_loss'              : train_loss['yaw_loss'],           \
                                     'train_iou_loss'              : train_loss['iou_loss'],           \
                                     'train_class_loss'            : train_loss['class_loss'],         \
                                     'train_depth_l2_loss'         : train_loss['depth_l2_loss'],      \
                                     'train_appearance_match_loss' : train_loss['depth_unsupervised_loss'],  \
                                     'train_depth_smooth_loss'     : train_loss['depth_smooth_loss'],  \
                                     'train_total_loss'            : train_loss['total_loss'],         \
                                     'valid_conf_loss'             : val_loss['conf_loss'],            \
                                     'valid_x_loss'                : val_loss['x_loss'],               \
                                     'valid_y_loss'                : val_loss['x_loss'],               \
                                     'valid_z_loss'                : val_loss['z_loss'],               \
                                     'valid_l_loss'                : val_loss['l_loss'],               \
                                     'valid_w_loss'                : val_loss['w_loss'],               \
                                     'valid_h_loss'                : val_loss['h_loss'],               \
                                     'valid_yaw_loss'              : val_loss['yaw_loss'],             \
                                     'valid_iou_loss'              : val_loss['iou_loss'],             \
                                     'valid_class_loss'            : val_loss['class_loss'],           \
                                     'valid_appearance_match_loss' : val_loss['depth_unsupervised_loss'],    \
                                     'valid_depth_l2_loss'         : val_loss['depth_l2_loss'],        \
                                     'valid_depth_smooth_loss'     : val_loss['depth_smooth_loss'],    \
                                     'valid_total_loss'            : val_loss['total_loss']             })

                # save model periodically
                if (epoch != 0) and (epoch % self.model_save_steps == 0):
                    model_name = os.path.join(model_exp_dir, 'model_epoch_'+str(epoch).zfill(4)+'.pt'.format(epoch))
                    print('Saving model: {}'.format(model_name))
                    torch.save(self.model.state_dict(), model_name)
                
                # save best model
                if train_mean_loss < min_train_mean_loss:
                    print('Loss improved from {:.4f} to {:.4f}. Saving model.'.format(min_train_mean_loss, train_mean_loss))
                    model_name = os.path.join(model_exp_dir, 'checkpoint_best.pt')
                    torch.save(self.model.state_dict(), model_name)
                    min_train_mean_loss = train_mean_loss
                    n_consecutive_epochs_no_improvement = 0
                # early stopping
                else:
                    n_consecutive_epochs_no_improvement += 1
                    print('Loss did not improve from {:.4f} for {} epoch[s].'.format(min_train_mean_loss, n_consecutive_epochs_no_improvement))
                    if n_consecutive_epochs_no_improvement >= self.early_stop_steps:
                        print('Early Stopping.')
                        break

    def predict(self, points, left_image, conf_thres=0.7):
        if (left_image.shape[-1] == 1) or (left_image.shape[-1] == 3):
            left_image = np.transpose(left_image, (-1, -3, -2))

        # normalize image
        left_image = normalize_img(left_image)
        left_image = np.expand_dims(left_image, 0).astype(np.float32)
        left_image = torch.from_numpy(np.asarray(left_image, dtype=np.float32)).to(device)

        # resize
        w, h = self.img_size # could be a batch or single image
        resize = transforms.Compose([
                            transforms.Resize(size=(h, w))
                        ])
        left_image = resize(left_image)

        # create mask
        points_mask = np.logical_and.reduce(((points[:,0] > self.xlim[0]), (points[:,0] < self.xlim[1]), \
                                              (points[:,1] > self.ylim[0]), (points[:,1] < self.ylim[1]), \
                                              (points[:,2] > self.zlim[0]), (points[:,2] < self.zlim[1])))

        # filter out
        points = points[:,:3] # only take x,y,z
        points = points[points_mask]

        # convert point-cloud to volume
        points_vol = point_cloud_to_volume(points, vol_size=self.vol_size, xlim=self.xlim, ylim=self.ylim, zlim=self.zlim)
        points_vol = np.expand_dims(points_vol, 0).astype(np.float32)
        points_vol = np.expand_dims(points_vol, 0).astype(np.float32)
        # put the volume in device
        points_vol = torch.from_numpy(points_vol).to(device)

        # model    
        self.model = self.model.to(device)
        self.model.eval()

        # forward pass
        with torch.no_grad():
            pred_tuple = self.model(points_vol, left_image)
            if self.dense_depth:
                pose_pred, depth_pred = pred_tuple
                depth_pred = depth_pred.cpu().numpy()
                depth_pred = denormalize_depth(depth_pred, self.xlim[1])
                depth_pred = np.squeeze(depth_pred[0], 0)
            else:
                pose_pred = pred_tuple

        label_pred = pose_pred.cpu().numpy()

        # get detections
        label_dict = decompose_label_vector(label_pred, self.n_xgrids, self.n_ygrids, \
                                    xlim=self.xlim, ylim=self.ylim, zlim=self.zlim, conf_thres=conf_thres)
        
        if self.dense_depth:
            return_tuple = (label_dict, depth_pred)
        else:
            return_tuple = label_dict

        # return predictions
        return return_tuple