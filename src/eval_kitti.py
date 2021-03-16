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
from models import *
from utils import *

# get device info
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# camera intrinsic matrix
K = np.array([[7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 4.485728000000e+01],
              [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 2.163791000000e-01],
              [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 2.745884000000e-03]])

# lidar-to-camera extrinsic matrix
T_lidar2cam = [[ 0.0002, -0.9999, -0.0106,  0.0594],
               [ 0.0104,  0.0106, -0.9999, -0.0751],
               [ 0.9999,  0.0001,  0.0105, -0.2721],
               [ 0.,      0.,      0.,      1.    ]]

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

    # define model
    obj_label_len = len(pose_fields) + len(label_map) # 9 for poses, rest for object classes
    model = VR3Dense(in_channels=1, n_xgrids=args.n_xgrids, n_ygrids=args.n_ygrids, obj_label_len=obj_label_len, \
                    dense_depth=args.dense_depth, train_depth_only=args.train_depth_only, train_obj_only=args.train_obj_only, \
                    concat_latent_vector=args.concat_latent_vector)
    model = model.to(device)

    # load weights
    model = load_pretrained_weights(model, args.modeldir, exp_str)
    
    # define trainer
    trainer = Trainer(dataroot=args.dataroot, model=model, dataset=None, mode='test', dense_depth=args.dense_depth, \
                      n_xgrids=args.n_xgrids, n_ygrids=args.n_ygrids, exp_str=exp_str, \
                      epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, \
                      xmin=args.xmin, xmax=args.xmax, ymin=args.ymin, ymax=args.ymax, zmin=args.zmin, zmax=args.zmax, \
                      max_depth=args.max_depth, vol_size_x=args.vol_size_x, vol_size_y=args.vol_size_y, vol_size_z=args.vol_size_z, \
                      img_size_x=args.img_size_x, img_size_y=args.img_size_y, loss_weights=[], \
                      modeldir=args.modeldir, logdir=args.logdir, plotdir=args.plotdir, \
                      model_save_steps=args.model_save_steps, early_stop_steps=args.early_stop_steps)

    # get a list of point-cloud bin files
    pc_filenames = sorted(glob.glob(os.path.join(args.pc_dir, '*.bin')))
    label_dict_list, depth_metrics_list, filenames = [], [], []

    # log time
    time_taken = []
    # iterate through all files
    for pc_filename in pc_filenames:

        # read point-cloud
        velo_pc = read_velo_bin(pc_filename)

        # read corresponding image
        fname, file_ext = os.path.splitext(pc_filename)
        fname = fname.split('/')[-1]
        img_fname = os.path.join(args.img_dir, fname+'.png')
        img_bgr = cv2.imread(img_fname)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # perform prediction
        pred_tuple, dt = trainer.predict(velo_pc, img_rgb)
        if args.dense_depth:
            label_dict, dense_depth = pred_tuple

            if args.eval_depth:
                projected_gt = project_pc2image(velo_pc, T_lidar2cam, K, (img_rgb.shape[1], img_rgb.shape[0]))
                dense_depth = cv2.resize(dense_depth, (img_rgb.shape[1], img_rgb.shape[0]), interpolation = cv2.INTER_NEAREST) 
                depth_metrics = compute_depth_metrics(projected_gt, dense_depth, max_depth=70.0)
                depth_metrics_list.append([depth_metrics['abs_rel'], 
                                        depth_metrics['sq_rel'], 
                                        depth_metrics['rmse'], 
                                        depth_metrics['rmse_log'], 
                                        depth_metrics['a1'], 
                                        depth_metrics['a2'], 
                                        depth_metrics['a3']])

        label_dict_list.append(label_dict)
        filenames.append(fname+'.txt')
        time_taken.append(dt)

        # print info
        print('\rInference done on: {}'.format(fname+'.bin/.png'), end='')
    print()

    if args.eval_object:
        # write to file
        predictions2file(label_dict_list, filenames, resolution=(1242, 375), K=K, exp=exp_id)

    if args.eval_depth:
        # print depth evaluation metrics
        depth_metrics_list = np.array(depth_metrics_list, dtype=np.float32)
        print('| ========================================================================== |')
        print('| abs_rel  | sq_rel   | rmse     | rmse_log | a1       | a2       | a3       |')
        print('| ========================================================================== |')
        print('| {:5f} | {:5f} | {:5f} | {:5f} | {:5f} | {:5f} | {:5f} |'.format(np.mean(depth_metrics_list[:,0]),
                                                                                np.mean(depth_metrics_list[:,1]),
                                                                                np.mean(depth_metrics_list[:,2]),
                                                                                np.mean(depth_metrics_list[:,3]),
                                                                                np.mean(depth_metrics_list[:,4]),
                                                                                np.mean(depth_metrics_list[:,5]),
                                                                                np.mean(depth_metrics_list[:,6])))
        print('| ========================================================================== |')
    # print timing
    time_taken_average = np.mean(time_taken)
    print('Tested on {} files | Average time taken per frame: {:.2f}ms | fps: {:.2f}'.format(
            len(pc_filenames), time_taken_average, 1000.0/time_taken_average))