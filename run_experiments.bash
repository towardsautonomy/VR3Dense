#!/bin/bash

# experiment mode
MODE=$1

# color codes
black=\e[0m
cyan=\e[96m
red=\e[91m

## run experiments
if [ "$MODE" == "train" ]; then
	# train
    python train.py --dataroot=/media/shubham/GoldMine/datasets/KITTI/object \
                    --epochs=100 --batch_size=8 --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=kitti \
                    --dense_depth --concat_latent_vector \
                    --use_pretrained_weight #--pretrained_weights=./models/vr3d.learning_rate_0.0001.n_xgrids_16.n_ygrids_16.xlim_0.0_70.0.ylim_-25.0_25.0.zlim_-2.5_1.0.max_depth_100.0.vol_size_256x256x16.img_size_512x256.dense_depth_True.exp_id_kitti/checkpoint_best.pt
  
    # # train for ablation study
    # python train.py --dataroot=/media/shubham/GoldMine/datasets/KITTI/object_subset \
    #                 --epochs=100 --batch_size=8 --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=kitti_no_latent_concat \
    #                 --lambda_depth_l2_loss=0 --lambda_depth_smooth_loss=0 \
    #                 --dense_depth
    
    # python train.py --dataroot=/media/shubham/GoldMine/datasets/KITTI/object_subset \
    #                 --epochs=100 --batch_size=8 --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=kitti_no_latent_concat_add_l2 \
    #                 --lambda_depth_smooth_loss=0 \
    #                 --dense_depth

    # python train.py --dataroot=/media/shubham/GoldMine/datasets/KITTI/object_subset \
    #                 --epochs=100 --batch_size=8 --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=kitti_no_latent_concat_add_l2_add_smooth \
    #                 --alpha_depth_smooth_loss=0 
    #                 --dense_depth

    # python train.py --dataroot=/media/shubham/GoldMine/datasets/KITTI/object_subset \
    #                 --epochs=100 --batch_size=8 --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=kitti_no_latent_concat_add_l2_add_edge_preserv_smooth \
    #                 --dense_depth

    # python train.py --dataroot=/media/shubham/GoldMine/datasets/KITTI/object_subset \
    #                 --epochs=100 --batch_size=8 --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=kitti_latent_concat_add_l2_add_edge_preserv_smooth \
    #                 --dense_depth --concat_latent_vector --use_pretrained_weight

elif [ "$MODE" == "test" ]; then
	# test
    python test.py --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --dense_depth --concat_latent_vector --exp_id=kitti 

elif [ "$MODE" == "test_and_compare" ]; then
	# test and compare
    python test_and_compare.py --dataroot=/media/shubham/GoldMine/datasets/KITTI/object \
                               --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --dense_depth --concat_latent_vector --exp_id=kitti 

elif [ "$MODE" == "evaluate" ]; then
	# evaluate
    python src/eval_kitti.py --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --dense_depth --concat_latent_vector --exp_id=kitti 

    # # evaluate for ablation study
    # python src/eval_kitti.py --epochs=100 --batch_size=8 --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=kitti_no_latent_concat \
    #                         --lambda_depth_l2_loss=0 --lambda_depth_smooth_loss=0 \
    #                         --dense_depth
    
    # python src/eval_kitti.py --epochs=100 --batch_size=8 --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=kitti_no_latent_concat_add_l2 \
    #                         --lambda_depth_smooth_loss=0 \
    #                         --dense_depth

    # python src/eval_kitti.py --epochs=100 --batch_size=8 --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=kitti_no_latent_concat_add_l2_add_smooth \
    #                         --dense_depth

    # python src/eval_kitti.py --epochs=100 --batch_size=8 --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=kitti_no_latent_concat_add_l2_add_edge_preserv_smooth \
    #                         --dense_depth

    # python src/eval_kitti.py --epochs=100 --batch_size=8 --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=kitti_latent_concat_add_l2_add_edge_preserv_smooth \
    #                         --dense_depth --concat_latent_vector
                    
else
    # error
    echo -e "\e[91mMode not provided. Please run this script with one of the following modes..
            \t ./run_experiments [train]
            \t ./run_experiments [test]
            \t ./run_experiments [evaluate]
            \t ./run_experiments [test_and_compare] \e[0m"
fi