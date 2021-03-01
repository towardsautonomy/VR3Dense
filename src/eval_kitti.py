# import modules
import torch
import glob
import sys
import os
# add path to sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.abspath('src'))
sys.path.append(BASE_DIR)
sys.path.append(SRC_DIR)

from config import parse_args
from trainer import Trainer
from datasets import KITTIObjectDataset
from models import *
from utils import *

# data path
test_data_path = '/media/shubham/GoldMine/datasets/KITTI/object/testing/velodyne'

# get device info
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# camera intrinsic matrix
K = np.array([[7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 4.485728000000e+01],
              [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 2.163791000000e-01],
              [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 2.745884000000e-03]])

# main function
if __name__ == "__main__":
    
    # parse arguments                
    args = parse_args()

    # experiment string
    exp_str = 'vr3d.epochs_{}.batch_size_{}.learning_rate_{}.n_xgrids_{}.n_ygrids_{}.xlim_{}_{}.ylim_{}_{}.zlim_{}_{}.vol_size_{}_{}_{}.exp_id_{}'.format(
                        args.epochs, args.batch_size, args.learning_rate, args.n_xgrids, args.n_ygrids, args.xmin, args.xmax, \
                        args.ymin, args.ymax, args.zmin, args.zmax, args.vol_size_x, args.vol_size_y, args.vol_size_z, args.exp_id)
    model_exp_dir = os.path.join(args.modeldir, exp_str)
    
    # define model
    obj_label_len = len(pose_fields) + len(label_map) # 9 for poses, rest for object classes
    model = VR3Dense(in_channels=1, n_xgrids=args.n_xgrids, n_ygrids=args.n_ygrids, obj_label_len=obj_label_len)
    model = model.to(device)

    # load weights
    model = load_pretrained_weights(model, args.modeldir, exp_str)
    
    # define trainer
    trainer = Trainer(data_dir=args.data_dir, model=model, dataset=None, \
                      n_xgrids=args.n_xgrids, n_ygrids=args.n_ygrids, exp_str=exp_str, \
                      epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, \
                      xmin=args.xmin, xmax=args.xmax, ymin=args.ymin, ymax=args.ymax, zmin=args.zmin, zmax=args.zmax, \
                      mode='test', vol_size_x=args.vol_size_x, vol_size_y=args.vol_size_y, vol_size_z=args.vol_size_z, \
                      modeldir=args.modeldir, logdir=args.logdir, plotdir=args.plotdir, \
                      model_save_steps=args.model_save_steps, early_stop_steps=args.early_stop_steps, test_split=0.0)

    # get a list of point-cloud bin files
    pc_filenames = sorted(glob.glob(os.path.join(test_data_path, '*.bin')))

    # iterate through all files
    label_dict_list = []
    filenames = [] 
    for i, pc_filename in enumerate(pc_filenames):
        fname, file_ext = os.path.splitext(pc_filename)
        fname = fname.split('/')[-1]
        # read point-cloud
        velo_pc = read_velo_bin(pc_filename)

        # perform prediction
        label_dict = trainer.predict(velo_pc)
        label_dict_list.append(label_dict)
        filenames.append(fname+'.txt')

        # print info
        print('Inference done on: {}'.format(fname+file_ext))

    # write to file
    predictions2file(label_dict_list, filenames, resolution=(1242, 375), K=K, exp='kitti_test')