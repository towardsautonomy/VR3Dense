# import modules
import os
import sys

# add path to sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.abspath('src'))
sys.path.append(BASE_DIR)
sys.path.append(SRC_DIR)

import torch
import glob
import csv
from prettytable import PrettyTable
from src import parse_args, Trainer
from src.datasets import KITTIObjectDataset
from src.models import *
from src.utils import *

# get device info
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class
obj_classes = ['Pedestrian', 'Cyclist']

# main function
if __name__ == "__main__":
    
    # parse arguments                
    args = parse_args()

    # experiment string
    exp_id = 'None'
    if args.exp_id != '':
        exp_id = args.exp_id
    exp_str = 'vr3d.learning_rate_{}.n_xgrids_{}.n_ygrids_{}.xlim_{}_{}.ylim_{}_{}.zlim_{}_{}.max_depth_{}.vol_size_{}x{}x{}.img_size_{}x{}.dense_depth_{}.concat_latent_vector_{}.exp_id_{}'.format(
                    args.learning_rate, args.n_xgrids, args.n_ygrids, args.xmin, args.xmax, args.ymin, args.ymax, \
                    args.zmin, args.zmax, args.max_depth, args.vol_size_x, args.vol_size_y, args.vol_size_z, args.img_size_x, \
                    args.img_size_y, args.dense_depth, args.concat_latent_vector, exp_id)
    model_exp_dir = os.path.join(args.modeldir, exp_str)

    # mean dimensions
    mean_lwh = {'Car':          args.car_mean_lwh, 
                'Cyclist':      args.cyclist_mean_lwh,
                'Pedestrian':   args.pedestrian_mean_lwh   }
                
    # define model
    obj_label_len = len(pose_fields) + len(label_map) # 9 for poses, rest for object classes
    model = VR3Dense(in_channels=1, n_xgrids=args.n_xgrids, n_ygrids=args.n_ygrids, obj_label_len=obj_label_len, \
                    dense_depth=args.dense_depth, train_depth_only=args.train_depth_only, train_obj_only=args.train_obj_only, \
                    concat_latent_vector=args.concat_latent_vector)
    model = model.to(device)

    # load weights
    model = load_pretrained_weights(model, args.modeldir, exp_str)
    
    # define trainer
    trainer = Trainer(dataroot=args.dataroot, model=model, dataset=KITTIObjectDataset, dense_depth=args.dense_depth, \
                      n_xgrids=args.n_xgrids, n_ygrids=args.n_ygrids, exp_str=exp_str, \
                      epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, \
                      xmin=args.xmin, xmax=args.xmax, ymin=args.ymin, ymax=args.ymax, zmin=args.zmin, zmax=args.zmax, \
                      max_depth=args.max_depth, vol_size_x=args.vol_size_x, vol_size_y=args.vol_size_y, vol_size_z=args.vol_size_z, \
                      img_size_x=args.img_size_x, img_size_y=args.img_size_y, loss_weights=[], mode='test', \
                      mean_lwh=mean_lwh, modeldir=args.modeldir, logdir=args.logdir, plotdir=args.plotdir, \
                      model_save_steps=args.model_save_steps, early_stop_steps=args.early_stop_steps, \
                      train_depth_only=args.train_depth_only, train_obj_only=args.train_obj_only)

    legend_list = []
    header = ['IOU Threshold', 'Conf Threshold', 'Precision', 'Recall']
    t = PrettyTable(header)

    # read input files
    poses_true_list, poses_pred_list = [], []
    for i in range(len(trainer.dataset)): 
        sample = trainer.dataset[i]
        poses_true_list.append(sample['label_dict'])
        # perform prediction
        pred_tuple, dt = trainer.predict(sample['cloud'], sample['left_image'])
        if args.dense_depth:
            label_dict, dense_depth = pred_tuple
        poses_pred_list.append(label_dict)
        print('\rInference done on frame: {}'.format(str(i).zfill(6)), end='')
    print()

    for obj_class in obj_classes:
        pr_csv_file = os.path.join(model_exp_dir, 'precision_recall_{}.csv'.format(obj_class))
        with open(pr_csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()

            for i, iou_thres in enumerate(np.arange(0.3, 0.8, 0.1)):
                precision, recall = [], []
                iou_thres_list=[]
                conf_thres_list=[]
                conf_thres_delta_stops = [0.01, 0.1, 0.9, 0.98, 0.999]
                conf_thres_delta_list = [0.02, 0.2, 0.1, 0.02, 0.01]
                conf_thres = conf_thres_delta_stops[0]
                conf_thres_delta = conf_thres_delta_list[0]

                while conf_thres < 1.0:
                    # compute precision and recall
                    p, r = compute_precision_recall(poses_true_list, poses_pred_list, iou_thres=iou_thres, conf_thres=conf_thres, classes=obj_class)
                    
                    precision.append(p)
                    recall.append(r)
                    iou_thres_list.append(iou_thres)
                    conf_thres_list.append(conf_thres)
                
                    # log to the csv file 'IOU Threshold', 'Conf Threshold', 'Precision', 'Recall'
                    writer.writerow({   'IOU Threshold' : iou_thres,  \
                                        'Conf Threshold': conf_thres, \
                                        'Precision'     : p,          \
                                        'Recall'        : r           })
                    t.add_row(['{:.3f}'.format(iou_thres), '{:.3f}'.format(conf_thres), '{:.3f}'.format(p), '{:.3f}'.format(r)])
                    print(t)

                    # increment threshold
                    if conf_thres >= conf_thres_delta_stops[0] and conf_thres < conf_thres_delta_stops[1]:
                        conf_thres_delta = conf_thres_delta_list[1]
                    elif conf_thres >= conf_thres_delta_stops[1] and conf_thres < conf_thres_delta_stops[2]:
                        conf_thres_delta = conf_thres_delta_list[2]
                    elif conf_thres >= conf_thres_delta_stops[2] and conf_thres < conf_thres_delta_stops[3]:
                        conf_thres_delta = conf_thres_delta_list[3]
                    elif conf_thres >= conf_thres_delta_stops[4] and conf_thres < conf_thres_delta_stops[4]:
                        conf_thres_delta = conf_thres_delta_list[4]
                    conf_thres += conf_thres_delta

                legend_list.append('IOU: {:.2f}'.format(iou_thres))