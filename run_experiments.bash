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
                    --epochs=500 --batch_size=16 --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=independent_obj_depth_models_subset \
                    --dense_depth \
                    --use_pretrained_weight #--pretrained_weights=./models/vr3d.learning_rate_0.0001.n_xgrids_16.n_ygrids_16.xlim_0.0_70.0.ylim_-25.0_25.0.zlim_-2.5_1.0.vol_size_256x256x16.img_size_256x256.dense_depth_True.exp_id_independent_obj_depth_models/checkpoint_best.pt
  
elif [ "$MODE" == "test" ]; then
	# test
    python test.py --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=None

elif [ "$MODE" == "test_and_compare" ]; then
	# test and compare
    python test_and_compare.py --dataroot=/media/shubham/GoldMine/datasets/KITTI/object \
                    --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=None 

elif [ "$MODE" == "evaluate" ]; then
	# evaluate
    python src/eval_kitti.py --dataroot=/media/shubham/GoldMine/datasets/KITTI/object \
                    --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=None 
                    
else
    # error
    echo -e "\e[91mMode not provided. Please run this script with one of the following modes..
            \t ./run_experiments [train]
            \t ./run_experiments [test]
            \t ./run_experiments [evaluate]
            \t ./run_experiments [test_and_compare] \e[0m"
fi