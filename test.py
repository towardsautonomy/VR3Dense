# import modules
import torch
import glob
import open3d as o3d
from src import parse_args, Trainer
from src.models import *
from src.utils import *
from src.AB3DMOT.AB3DMOT_libs.model import AB3DMOT

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

# canvas size
canvasSize = 800

## config parameters
# whether or not to display point-cloud
disp_cloud = False
# tracker
TRACKING = True
# write frames to file
WRITE_TO_FILE = False
OUT_DIR = 'tmp/vr3dense_demo'

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

    # visualization window
    cv2.namedWindow('VR3Dense', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('VR3Dense', 900, 1440)
    pcd = o3d.geometry.PointCloud()
    vis = None
    if disp_cloud == True:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
    # cloud viewpoint params
    angle_diff_range = list(np.arange(-7.0, 20.0, 1.0)) + list(np.arange(20.0, 0.0, -1.0)) + [0.0]*50
    zoom_diff_range = list(np.arange(0.2, 0.257, 0.001)) + list(np.arange(0.257, 0.2, -0.005))
    # iterate through all files
    for idx, pc_filename in enumerate(pc_filenames):
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

        # labels
        if TRACKING == True:
            label_dict = label_tracked_dict

        # get labels in camera coordinate system
        label_cam = label_lidar2cam(label_dict, T_lidar2cam)
        # draw bounding box on image
        img_rgb_bbox = draw_bbox_img(img_rgb.copy(), label_cam, K)

        # resize image
        scale_factor = canvasSize / img_rgb.shape[1] 
        width = int(img_rgb.shape[1] * scale_factor)
        height = int(img_rgb.shape[0] * scale_factor)
        img_rgb_resized = cv2.resize(img_rgb_bbox, (width, height), interpolation = cv2.INTER_AREA) 
        img_rgb_resized = np.array(img_rgb_resized, dtype=np.uint8)

        # predicted depth
        if args.dense_depth:
            dense_depth = cv2.resize(dense_depth, (img_rgb.shape[1], img_rgb.shape[0]), interpolation = cv2.INTER_NEAREST) 
            dense_depth_colorized = colorize_depth_map(dense_depth)
            dense_depth_colorized = cv2.resize(dense_depth_colorized, (width, height), interpolation = cv2.INTER_NEAREST) 
            dense_depth_colorized = cv2.cvtColor(dense_depth_colorized, cv2.COLOR_RGB2BGR)
            dense_depth_colorized = np.array(dense_depth_colorized, dtype=np.uint8)

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
        img_bgr = cv2.cvtColor(img_rgb_resized, cv2.COLOR_RGB2BGR)

        if args.dense_depth:
            img_viz = cv2.vconcat([img_bgr, dense_depth_colorized, pc_bbox_img_bgr])
        else:
            img_viz = cv2.vconcat([img_bgr, pc_bbox_img_bgr])
        # show image
        cv2.imshow('VR3Dense', img_viz)
        cv2.waitKey(1)

        # build a point-cloud
        pts3d = []
        pts3dcolor = []

        if disp_cloud == True:
            # point-cloud
            for i in range(dense_depth.shape[0]):
                for j in range(dense_depth.shape[1]):
                    z = dense_depth[i,j]
                    if z <100.0:
                        x = (j - K[0,2]) * z / K[0,0]
                        y = (i - K[1,2]) * z / K[1,1]
                        r = img_rgb[i,j,0]
                        g = img_rgb[i,j,1]
                        b = img_rgb[i,j,2]
                        pts3d.append([x, y, z])
                        pts3dcolor.append([r, g, b])

            pts3d = np.array(pts3d)
            pts3dcolor = np.array(pts3dcolor, np.float32) / 255.0

            # add to open3d object
            pcd.points = o3d.utility.Vector3dVector(pts3d)
            pcd.colors = o3d.utility.Vector3dVector(pts3dcolor)
            pcd.rotate(pcd.get_rotation_matrix_from_xyz((np.pi+np.radians(angle_diff_range[idx % len(angle_diff_range)]), 0., 0.)))
            # o3d.visualization.draw_geometries([pcd])
            if idx == 0:
                vis.add_geometry(pcd)
                vis.get_view_control().set_zoom(0.2)
            else:
                vis.update_geometry(pcd)
                vis.get_view_control().set_zoom(zoom_diff_range[idx % len(zoom_diff_range)])
            vis.poll_events()
            vis.update_renderer()
        
        # write to file
        if WRITE_TO_FILE == True:
            if not os.path.exists(OUT_DIR):
                os.system('mkdir -p {}'.format(OUT_DIR))

            out_fname = os.path.join(OUT_DIR, fname+'.png')
            cv2.imwrite(out_fname, img_viz)

            if disp_cloud == True:
                if not os.path.exists(os.path.join(OUT_DIR, 'rgbd')):
                    os.system('mkdir -p {}'.format(os.path.join(OUT_DIR, 'rgbd')))
                rgbd_vis_fname = os.path.join(OUT_DIR, 'rgbd', fname+'.png')
                vis.capture_screen_image(rgbd_vis_fname)

            print('wrote frame to: {}'.format(out_fname))