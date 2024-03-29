# import modules
import torch
from torchsummary import summary
import hiddenlayer as hl
import yaml
from src import parse_args, Trainer
from src.datasets import KITTIObjectDataset
from src.models import *
from src.utils import *

# get device info
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# count number of parameters in a model
def count_model_parameters(model):
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_non_trainable_params = n_params - n_trainable_params

    return n_params, n_trainable_params, n_non_trainable_params

# main function
if __name__ == "__main__":
    
    # parse arguments                
    args = parse_args()

    # print arguments
    vr3dense_args = {}
    print('==========================================================')
    print('Training with the following parameters:')
    print('==========================================================')
    for arg in vars(args):
        print('{0:>25}: '.format(arg), end='')
        print('{}'.format(getattr(args, arg)))
        vr3dense_args[arg] = getattr(args, arg)
    print('==========================================================')

    # experiment string
    exp_id = 'None'
    if args.exp_id != '':
        exp_id = args.exp_id
    exp_str = 'vr3d.learning_rate_{}.n_xgrids_{}.n_ygrids_{}.xlim_{}_{}.ylim_{}_{}.zlim_{}_{}.max_depth_{}.vol_size_{}x{}x{}.img_size_{}x{}.dense_depth_{}.concat_latent_vector_{}.exp_id_{}'.format(
                    args.learning_rate, args.n_xgrids, args.n_ygrids, args.xmin, args.xmax, args.ymin, args.ymax, \
                    args.zmin, args.zmax, args.max_depth, args.vol_size_x, args.vol_size_y, args.vol_size_z, args.img_size_x, \
                    args.img_size_y, args.dense_depth, args.concat_latent_vector, exp_id)
    model_exp_dir = os.path.join(args.modeldir, exp_str)
    # make directories
    os.system('mkdir -p {}'.format(model_exp_dir))
    
    # mean dimensions
    mean_lwh = {'Car':          args.car_mean_lwh, 
                'Cyclist':      args.cyclist_mean_lwh,
                'Pedestrian':   args.pedestrian_mean_lwh   }
                
    # lambda weights
    loss_weights = [args.lambda_conf_loss, 
                      args.lambda_x_loss,
                      args.lambda_y_loss,
                      args.lambda_z_loss,
                      args.lambda_l_loss,
                      args.lambda_w_loss,
                      args.lambda_h_loss,
                      args.lambda_yaw_loss,
                      args.lambda_iou_loss,
                      args.lambda_class_loss,
                      args.lambda_depth_unsup_loss,
                      args.lambda_depth_l2_loss,
                      args.lambda_depth_smooth_loss,
                      args.alpha_depth_smooth_loss]

    # define model
    obj_label_len = len(pose_fields) + len(label_map) # 9 for poses, rest for object classes
    model = VR3Dense(in_channels=1, n_xgrids=args.n_xgrids, n_ygrids=args.n_ygrids, obj_label_len=obj_label_len, \
                    dense_depth=args.dense_depth, train_depth_only=args.train_depth_only, train_obj_only=args.train_obj_only, \
                    concat_latent_vector=args.concat_latent_vector)
    model = model.to(device)
    
    # print model summary
    print('==========================================================')
    print('=====================  Model Summary  ====================')
    print('==========================================================')
    n_params, n_trainable_params, n_non_trainable_params = count_model_parameters(model)
    print('\t- Num of Parameters                : {:,}'.format(n_params))
    print('\t- Num of Trainable Parameters      : {:,}'.format(n_trainable_params))
    print('\t- Num of Non-Trainable Parameters  : {:,}'.format(n_non_trainable_params))
    print('==========================================================')
    # summary(model, [(1,args.vol_size_z,args.vol_size_x,args.vol_size_y), (3,args.img_size_x,args.img_size_y)])

    # write the dot graph to file
    dummy_x_lidar = torch.randn(1,1,args.vol_size_z,args.vol_size_x,args.vol_size_y).to(device)
    dummy_x_camera = torch.randn(1,3,args.img_size_x,args.img_size_y).to(device)
    hl_graph = hl.build_graph(model, (dummy_x_lidar, dummy_x_camera))
    hl_graph = hl_graph.save(os.path.join(model_exp_dir, 'model'))
    # write training hyperparameters to yaml file
    with open(os.path.join(model_exp_dir, 'vr3dense_args.yaml'), 'w') as outfile:
        yaml.dump(vr3dense_args, outfile, default_flow_style=False)

    # load weights
    best_ckpt_model = os.path.join(model_exp_dir, 'checkpoint_best.pt')
    if (args.use_pretrained_weights == True) and (args.pretrained_weights != 'none') and os.path.exists(args.pretrained_weights):
        model.load_state_dict(torch.load(args.pretrained_weights, map_location=lambda storage, loc: storage))
        print('Loaded pre-trained weights: {}'.format(args.pretrained_weights))
    elif (args.use_pretrained_weights == True) and os.path.exists(best_ckpt_model):
        model.load_state_dict(torch.load(best_ckpt_model, map_location=lambda storage, loc: storage))
        print('Loaded pre-trained weights: {}'.format(best_ckpt_model))
    elif (args.use_pretrained_weights == True):
        print('Pre-trained weights not found.')
    
    # define trainer
    trainer = Trainer(dataroot=args.dataroot, model=model, dataset=KITTIObjectDataset, dense_depth=args.dense_depth, \
                      n_xgrids=args.n_xgrids, n_ygrids=args.n_ygrids, exp_str=exp_str, \
                      epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, \
                      xmin=args.xmin, xmax=args.xmax, ymin=args.ymin, ymax=args.ymax, zmin=args.zmin, zmax=args.zmax, \
                      max_depth=args.max_depth, vol_size_x=args.vol_size_x, vol_size_y=args.vol_size_y, vol_size_z=args.vol_size_z, \
                      img_size_x=args.img_size_x, img_size_y=args.img_size_y, loss_weights=loss_weights, \
                      mean_lwh=mean_lwh, modeldir=args.modeldir, logdir=args.logdir, plotdir=args.plotdir, \
                      model_save_steps=args.model_save_steps, early_stop_steps=args.early_stop_steps, \
                      train_depth_only=args.train_depth_only, train_obj_only=args.train_obj_only)

    # train the model
    trainer.train()