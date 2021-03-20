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
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from prettytable import PrettyTable
from src import parse_args
from src.utils import *

# object class
obj_classes = ['Car', 'Pedestrian', 'Cyclist']

# N-point interpolation for mAP computation
def mAP(precision, recall, n_recall_pts=41):
    delta_n = 1.0/float(n_recall_pts-1)
    recall_stops = np.arange(0.0, 1.0, delta_n)
    
    # sort recall in increasing order
    recall_sort_indices = np.argsort(recall)
    recall = np.array(recall)[recall_sort_indices]
    precision = np.array(precision)[recall_sort_indices]

    precision_npts = []
    recall_npts = []
    for recall_ in recall_stops:
        recall_diff = recall - recall_
        recall_diff[recall_diff < 0.] = 1e6
        idx_recall = np.argmin(recall_diff)
        if recall_diff[idx_recall] > n_recall_pts:
            precision_ = 0.
        else:
            precision_ = np.max(precision[idx_recall:])
        recall_npts.append(recall_)
        precision_npts.append(precision_)

    # compute mean
    mAP_ = np.mean(precision_npts)
    return np.array(precision_npts), np.array(recall_npts), mAP_

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

    for obj_class in obj_classes:
        pr_csv_file = os.path.join(model_exp_dir, 'precision_recall_{}.csv'.format(obj_class))
        
        # assemble values
        iou_thres, conf_thres, precision, recall = [], [], [], []
        iou_thres_list, conf_thres_list, precision_list, recall_list = [], [], [], []
        legend_list = []
        prev_iou_thres = 0.0

        # figure to plot precision-recall curve
        plt.figure()
        plt.grid(linestyle='-', linewidth='0.2', color='gray')
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.1)

        # read the log file
        with open(pr_csv_file) as file:
            reader = csv.DictReader( file )
            for line in reader:
                iou_thres_ = float(line['IOU Threshold'])
                if prev_iou_thres != iou_thres_:
                    if len(iou_thres) > 0:
                        # append to the list of lists
                        iou_thres_list.append(iou_thres)
                        conf_thres_list.append(conf_thres)
                        precision_list.append(precision)
                        recall_list.append(recall)

                    # reset the values
                    iou_thres, conf_thres, precision, recall = [], [], [], []

                # append to the list
                iou_thres.append(iou_thres_)
                conf_thres.append(float(line['Conf Threshold']))
                precision.append(float(line['Precision']))
                recall.append(float(line['Recall']))
                prev_iou_thres = iou_thres_
            
            if len(iou_thres) > 0:
                iou_thres_list.append(iou_thres)
                conf_thres_list.append(conf_thres)
                precision_list.append(precision)
                recall_list.append(recall)

        table = PrettyTable(['IOU Threshold', 'mAP'])
        mAP_list = []
        for i, iou_thres in enumerate(iou_thres_list):
            conf_thres = conf_thres_list[i]
            precision = precision_list[i]
            recall = recall_list[i]
            precision_, recall_, mAP_ = mAP(precision, recall)
            mAP_list.append(mAP_)
            if iou_thres[0] > 0.8:
                continue
            plt.plot(recall_, precision_, '-')
            legend_list.append('IOU: {:.2f}'.format(iou_thres[0]))
            table.add_row(['{:.3f}'.format(iou_thres[0]), '{:.3f}'.format(mAP_)])

        print('mAP for class: {}'.format(obj_class))
        print(table)
        plt.title('Precision-Recall Curve')
        plt.legend(legend_list, loc='upper left', fancybox=True, framealpha=1., shadow=True, borderpad=1)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig(os.path.join(model_exp_dir, 'precision_recall_{}.png'.format(obj_class)))