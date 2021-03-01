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
    exp_str = 'vr3d.learning_rate_{}.n_xgrids_{}.n_ygrids_{}.xlim_{}_{}.ylim_{}_{}.zlim_{}_{}.vol_size_{}_{}_{}.exp_id_{}'.format(
                    args.learning_rate, args.n_xgrids, args.n_ygrids, args.xmin, args.xmax, args.ymin, args.ymax, \
                    args.zmin, args.zmax, args.vol_size_x, args.vol_size_y, args.vol_size_z, exp_id)
    model_exp_dir = os.path.join(args.modeldir, exp_str)
    
    # define model
    obj_label_len = len(pose_fields) + len(label_map) # 9 for poses, rest for object classes
    model = VR3Dense(in_channels=1, n_xgrids=args.n_xgrids, n_ygrids=args.n_ygrids, obj_label_len=obj_label_len)
    model = model.to(device)

    # load weights
    best_ckpt_model = os.path.join(model_exp_dir, 'checkpoint_best.pt')
    if (args.pretrained_weights != 'none') and os.path.exists(args.pretrained_weights):
        model.load_state_dict(torch.load(args.pretrained_weights, map_location=lambda storage, loc: storage))
        print('Loaded pre-trained weights: {}'.format(args.pretrained_weights))
    elif os.path.exists(best_ckpt_model):
        model.load_state_dict(torch.load(best_ckpt_model, map_location=lambda storage, loc: storage))
        print('Loaded pre-trained weights: {}'.format(best_ckpt_model))
    else:
        raise Exception('Pre-trained weights not found.')
    
    # define trainer
    trainer = Trainer(dataroot=args.dataroot, model=model, dataset=KITTIObjectDataset, \
                      n_xgrids=args.n_xgrids, n_ygrids=args.n_ygrids, exp_str=exp_str, \
                      epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, \
                      xmin=args.xmin, xmax=args.xmax, ymin=args.ymin, ymax=args.ymax, zmin=args.zmin, zmax=args.zmax, \
                      vol_size_x=args.vol_size_x, vol_size_y=args.vol_size_y, vol_size_z=args.vol_size_z, \
                      modeldir=args.modeldir, logdir=args.logdir, plotdir=args.plotdir, \
                      model_save_steps=args.model_save_steps, early_stop_steps=args.early_stop_steps)

    # get a list of point-cloud bin files
    dataset = KITTIObjectDataset(dataroot=args.dataroot, \
                                 n_xgrids=args.n_xgrids, n_ygrids=args.n_xgrids)

    # show 100 samples
    cv2.namedWindow('VR3Dense', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('VR3Dense', 800, 1440)
    for i in range(100): 
        sample = dataset[i]
        ## get true labels visualization
        pc_bbox_img_true = draw_point_cloud_w_bbox(sample['cloud'], sample['label_dict'], \
                                                    xlim=trainer.xlim, ylim=trainer.ylim, zlim=trainer.zlim)
        pc_bbox_img_true_bgr = cv2.cvtColor(pc_bbox_img_true, cv2.COLOR_RGB2BGR)

        ## get predicted labels visualization
        # perform prediction
        label_dict = trainer.predict(sample['cloud'])

        # get visualization
        pc_bbox_img_pred = draw_point_cloud_w_bbox(sample['cloud'], label_dict, \
                                                    xlim=trainer.xlim, ylim=trainer.ylim, zlim=trainer.zlim)
        pc_bbox_img_pred_bgr = cv2.cvtColor(pc_bbox_img_pred, cv2.COLOR_RGB2BGR)

        # visualization image
        img_viz = cv2.vconcat([pc_bbox_img_true_bgr, pc_bbox_img_pred_bgr])
        cv2.line(img_viz, (0,pc_bbox_img_true_bgr.shape[0]), (pc_bbox_img_true_bgr.shape[1]-1,pc_bbox_img_true_bgr.shape[0]), color=(255,255,255), thickness=2)
        cv2.imshow('VR3Dense', img_viz)
        cv2.waitKey(0)