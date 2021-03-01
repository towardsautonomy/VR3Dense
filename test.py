# import modules
import torch
import glob
from src import parse_args, Trainer
from src.datasets import KITTIObjectDataset
from src.models import *
from src.utils import *
from src.AB3DMOT.AB3DMOT_libs.model import AB3DMOT

# data path
test_pc_path = '/media/shubham/GoldMine/datasets/KITTI/raw/2011_09_26/2011_09_26_drive_0009_sync/velodyne_points/data'
test_img_path = '/media/shubham/GoldMine/datasets/KITTI/raw/2011_09_26/2011_09_26_drive_0009_sync/image_02/data'

# get device info
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# camera intrinsic matrix
K = np.array([[7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 4.485728000000e+01],
              [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 2.163791000000e-01],
              [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 2.745884000000e-03]])

# lidar-to-camera extrinsics
T_lidar2cam = np.array([[ 0.0002, -0.9999, -0.0106,  0.0594],
                        [ 0.0104,  0.0106, -0.9999, -0.0751],
                        [ 0.9999,  0.0001,  0.0105, -0.2721],
                        [ 0.,      0.,      0.,      1.    ]])

canvasSize = 1200

## config parameters
# tracker
TRACKING = True
# write frames to file
WRITE_TO_FILE = False
OUT_DIR = 'tmp/demo'

# main function
if __name__ == "__main__":
    
    # parse arguments                
    args = parse_args()

    # create an instance of tracker
    mot_tracker = AB3DMOT(max_age=2, min_hits=2)

    # experiment string
    exp_id = 'None'
    if args.exp_id != '':
        exp_id = args.exp_id
    exp_str = 'vr3d.learning_rate_{}.n_xgrids_{}.n_ygrids_{}.xlim_{}_{}.ylim_{}_{}.zlim_{}_{}.vol_size_{}_{}_{}.exp_id_{}'.format(
                    args.learning_rate, args.n_xgrids, args.n_ygrids, args.xmin, args.xmax, args.ymin, args.ymax, \
                    args.zmin, args.zmax, args.vol_size_x, args.vol_size_y, args.vol_size_z, exp_id)
    model_exp_dir = os.path.join(args.modeldir, exp_str)
    
    # define model
    obj_label_len = len(pose_fields) + len(label_map) # 9 for poses, rest for object classes
    model = VR3Dense(in_channels=1, n_xgrids=args.n_xgrids, n_ygrids=args.n_ygrids, obj_label_len=obj_label_len)
    model = model.to(device)

    # load weights
    model = load_pretrained_weights(model, args.modeldir, exp_str)
    
    # define trainer
    trainer = Trainer(dataroot=args.dataroot, model=model, dataset=None, \
                      n_xgrids=args.n_xgrids, n_ygrids=args.n_ygrids, exp_str=exp_str, \
                      epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, \
                      xmin=args.xmin, xmax=args.xmax, ymin=args.ymin, ymax=args.ymax, zmin=args.zmin, zmax=args.zmax, \
                      mode='test',vol_size_x=args.vol_size_x, vol_size_y=args.vol_size_y, vol_size_z=args.vol_size_z, \
                      modeldir=args.modeldir, logdir=args.logdir, plotdir=args.plotdir, \
                      model_save_steps=args.model_save_steps, early_stop_steps=args.early_stop_steps)

    # get a list of point-cloud bin files
    pc_filenames = sorted(glob.glob(os.path.join(test_pc_path, '*.bin')))

    # visualization window
    cv2.namedWindow('VR3Dense', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('VR3Dense', 900, 1440)
    # iterate through all files
    for pc_filename in pc_filenames:
        # read point-cloud
        velo_pc = read_velo_bin(pc_filename)

        # perform prediction
        label_dict = trainer.predict(velo_pc)

        # tracking
        label_tracked_dict = []
        if TRACKING == True:
            # get detection list for mot_tracker
            mot_det = []
            mot_other_info = []
            for label_ in label_dict:
                det = [label_['h'], label_['w'], label_['l'], label_['x'], label_['y'], label_['z'], label_['yaw']]
                other_info = (label_['class'], label_['conf'])
                mot_det.append(det)
                mot_other_info.append(other_info)

            mot_det = np.array(mot_det)
            mot_other_info = np.array(mot_other_info)
            # prepare the input of tracker
            dets_all = {'dets': mot_det, 'info': mot_other_info}

            # update the tracker
            trackers = []
            tracked_poses = []
            id_list = []
            if mot_det.shape[0] > 0:
                trackers = mot_tracker.update(dets_all)
                # extract output of tracker
                for d in trackers:
                    label_tracked_ = {}
                    h, w, l, x, y, z, yaw, id_, class_, conf = d
                    label_tracked_['h'], label_tracked_['w'], label_tracked_['l'], \
                        label_tracked_['x'], label_tracked_['y'], label_tracked_['z'], label_tracked_['yaw'], \
                        label_tracked_['id'], label_tracked_['class'], label_tracked_['conf'] = \
                            float(h), float(w), float(l), float(x), float(y), float(z), float(yaw), int(id_), str(class_), float(conf)
                    label_tracked_dict.append(label_tracked_)

            # perform non-max suppression to suppress duplicate tracklets
            label_tracked_dict = non_max_suppression(label_tracked_dict)

        # read corresponding image
        fname, file_ext = os.path.splitext(pc_filename)
        fname = fname.split('/')[-1]
        img_fname = os.path.join(test_img_path, fname+'.png')
        img_bgr = cv2.imread(img_fname)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # labels
        if TRACKING == True:
            label_dict = label_tracked_dict

        # get labels in camera coordinate system
        label_cam = label_lidar2cam(label_dict, T_lidar2cam)
        # draw bounding box on image
        img_rgb = draw_bbox_img(img_rgb, label_cam, K)

        # resize image
        scale_factor = canvasSize / img_rgb.shape[1] 
        width = int(img_rgb.shape[1] * scale_factor)
        height = int(img_rgb.shape[0] * scale_factor)
        img_rgb = cv2.resize(img_rgb, (width, height), interpolation = cv2.INTER_AREA) 
        img_rgb = np.array(img_rgb, dtype=np.uint8)

        # get visualization
        if TRACKING == True:
            pc_bbox_img = draw_point_cloud_w_bbox_id(velo_pc, label_dict, canvasSize=canvasSize, \
                                                        xlim=trainer.xlim, ylim=trainer.ylim, zlim=trainer.zlim)
        else:
            pc_bbox_img = draw_point_cloud_w_bbox(velo_pc, label_dict, canvasSize=canvasSize, \
                                                        xlim=trainer.xlim, ylim=trainer.ylim, zlim=trainer.zlim)

        pc_bbox_img_bgr = cv2.cvtColor(pc_bbox_img, cv2.COLOR_RGB2BGR)
        pc_bbox_img_bgr = np.array(pc_bbox_img_bgr*255.0, dtype=np.uint8)
        # concat image with point-cloud 
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_viz = cv2.vconcat([img_bgr, pc_bbox_img_bgr])
        # show image
        cv2.imshow('VR3Dense', img_viz)
        cv2.waitKey(1)
        
        # write to file
        if WRITE_TO_FILE == True:
            if not os.path.exists(OUT_DIR):
                os.system('mkdir -p {}'.format(OUT_DIR))
            out_fname = os.path.join(OUT_DIR, fname+'.png')
            cv2.imwrite(out_fname, img_viz)
            print('wrote frame to: {}'.format(out_fname))