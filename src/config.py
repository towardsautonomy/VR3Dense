import argparse

def parse_args():
    """
    This function parses the input arguments
    """
    # set up input arguments 
    parser = argparse.ArgumentParser(description='Setup parameters as arguments')
    parser.add_argument('--dataroot', type=str, default='./data',
                        help='data root directory')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning_rate')
    parser.add_argument('--n_xgrids', type=int, default=16,
                        help='number of grids in x direction')
    parser.add_argument('--n_ygrids', type=int, default=16,
                        help='number of grids in y direction')
    parser.add_argument('--modeldir', type=str, default='./models',
                        help='directory to save models')
    parser.add_argument('--logdir', type=str, default='./logs',
                        help='logs directory for generating CSV and tensorboard logs')
    parser.add_argument('--plotdir', type=str, default='./plots',
                        help='directory for generating plots')
    parser.add_argument('--model_save_steps', type=int, default=20,
                        help='number of steps to save the model')
    parser.add_argument('--early_stop_steps', type=int, default=20,
                        help='number of steps to wait for before stopping the training process if model does not improve')
    parser.add_argument('--use_pretrained_weights', default=False, action='store_true',
                        help='whether or not to use pretrained weights')
    parser.add_argument('--pretrained_weights', type=str, default='none',
                        help='pretrained weights file to use')
    parser.add_argument('--xmin', type=float, default=0.0,
                        help='minimum value of x to be considered for object detection')
    parser.add_argument('--xmax', type=float, default=70.0,
                        help='maximum value of x to be considered for object detection')
    parser.add_argument('--ymin', type=float, default=-25.0,
                        help='minimum value of y to be considered for object detection')
    parser.add_argument('--ymax', type=float, default=25.0,
                        help='maximum value of y to be considered for object detection')
    parser.add_argument('--zmin', type=float, default=-2.5,
                        help='minimum value of z to be considered for object detection')
    parser.add_argument('--zmax', type=float, default=1.0,
                        help='maximum value of z to be considered for object detection')
    parser.add_argument('--max_depth', type=float, default=100.0,
                        help='maximum depth value for depth prediction')
    parser.add_argument('--vol_size_x', type=int, default=256,
                        help='volume size for voxelizing point-cloud in x direction')
    parser.add_argument('--vol_size_y', type=int, default=256,
                        help='volume size for voxelizing point-cloud in y direction')
    parser.add_argument('--vol_size_z', type=int, default=16,
                        help='volume size for voxelizing point-cloud in z direction')
    parser.add_argument('--img_size_x', type=int, default=512,
                        help='image size to be resized in x direction')
    parser.add_argument('--img_size_y', type=int, default=256,
                        help='image size to be resized in y direction')
    parser.add_argument('--lambda_conf_loss', type=float, default=1.0,
                        help='lambda weight for conf loss')
    parser.add_argument('--lambda_x_loss', type=float, default=1.0,
                        help='lambda weight for x loss')
    parser.add_argument('--lambda_y_loss', type=float, default=1.0,
                        help='lambda weight for y loss')
    parser.add_argument('--lambda_z_loss', type=float, default=1.0,
                        help='lambda weight for z loss')
    parser.add_argument('--lambda_l_loss', type=float, default=1.0,
                        help='lambda weight for l loss')
    parser.add_argument('--lambda_w_loss', type=float, default=1.0,
                        help='lambda weight for w loss')
    parser.add_argument('--lambda_h_loss', type=float, default=1.0,
                        help='lambda weight for h loss')
    parser.add_argument('--lambda_yaw_loss', type=float, default=2.0,
                        help='lambda weight for yaw loss')
    parser.add_argument('--lambda_iou_loss', type=float, default=0.2,
                        help='lambda weight for giou loss')
    parser.add_argument('--lambda_class_loss', type=float, default=0.2,
                        help='lambda weight for class loss')
    parser.add_argument('--lambda_depth_unsup_loss', type=float, default=1.0,
                        help='lambda weight for depth unsupervised loss')
    parser.add_argument('--lambda_depth_l2_loss', type=float, default=0.5,
                        help='lambda weight for depth l2 loss')
    parser.add_argument('--lambda_depth_smooth_loss', type=float, default=0.1,
                        help='lambda weight for depth edge preserving smoothness loss')
    parser.add_argument('--alpha_depth_smooth_loss', type=float, default=0.5,
                        help='alpha for depth edge preserving smoothness loss')
    parser.add_argument('--dense_depth', default=False, action='store_true',
                        help='whether or not to use RGB-to-dense depth prediction network')
    parser.add_argument('--concat_latent_vector', default=False, action='store_true',
                        help='whether or not to concat latent vectors of object detection and depth prediction network')
    parser.add_argument('--train_depth_only', default=False, action='store_true',
                        help='whether or not to train only the depth prediction network')
    parser.add_argument('--train_obj_only', default=False, action='store_true',
                        help='whether or not to train only the object detection network')
    parser.add_argument('--exp_id', type=str, default='None',
                        help='experiment identifier')

    return parser.parse_args()