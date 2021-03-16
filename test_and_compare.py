# import modules
import torch
import glob
from src import parse_args, Trainer
from src.datasets import KITTIObjectDataset
from src.models import *
from src.utils import *

# get device info
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    trainer = Trainer(dataroot=args.dataroot, model=model, dataset=KITTIObjectDataset, dense_depth=args.dense_depth, \
                      n_xgrids=args.n_xgrids, n_ygrids=args.n_ygrids, exp_str=exp_str, \
                      epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, \
                      xmin=args.xmin, xmax=args.xmax, ymin=args.ymin, ymax=args.ymax, zmin=args.zmin, zmax=args.zmax, \
                      max_depth=args.max_depth, vol_size_x=args.vol_size_x, vol_size_y=args.vol_size_y, vol_size_z=args.vol_size_z, \
                      img_size_x=args.img_size_x, img_size_y=args.img_size_y, loss_weights=[], \
                      modeldir=args.modeldir, logdir=args.logdir, plotdir=args.plotdir, \
                      model_save_steps=args.model_save_steps, early_stop_steps=args.early_stop_steps, \
                      train_depth_only=args.train_depth_only, train_obj_only=args.train_obj_only)

    # show 100 samples
    cv2.namedWindow('VR3Dense', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('VR3Dense', 800, 1440)
    for i in range(100): 
        sample = trainer.dataset[i]
        ## get true labels visualization
        pc_bbox_img_true = draw_point_cloud_w_bbox(sample['cloud'], sample['label_dict'], \
                                                    xlim=trainer.xlim, ylim=trainer.ylim, zlim=trainer.zlim)
        pc_bbox_img_true_bgr = cv2.cvtColor(pc_bbox_img_true, cv2.COLOR_RGB2BGR)

        ## get predicted labels visualization
        # perform prediction
        pred_tuple, dt = trainer.predict(sample['cloud'], sample['left_image'])
        if args.dense_depth:
            label_dict, dense_depth = pred_tuple

        # get visualization
        pc_bbox_img_pred = draw_point_cloud_w_bbox(sample['cloud'], label_dict, \
                                                    xlim=trainer.xlim, ylim=trainer.ylim, zlim=trainer.zlim)
        pc_bbox_img_pred_bgr = cv2.cvtColor(pc_bbox_img_pred, cv2.COLOR_RGB2BGR)

        # visualization image
        img_viz = cv2.vconcat([pc_bbox_img_true_bgr, pc_bbox_img_pred_bgr])
        cv2.line(img_viz, (0,pc_bbox_img_true_bgr.shape[0]), (pc_bbox_img_true_bgr.shape[1]-1,pc_bbox_img_true_bgr.shape[0]), color=(255,255,255), thickness=2)

        cv2.imshow('VR3Dense', img_viz)
        cv2.waitKey(0)