#!/bin/bash

# experiment mode
MODE=$1
OPTION="none"
if [ "$#" -eq 2 ]; then
    OPTION=$2
fi

# color codes
black=\e[0m
cyan=\e[96m
red=\e[91m

## run experiments
if [ "$MODE" == "train" ]; then

    if [ "$OPTION" == "none" ]; then
        # train
        python train.py --dataroot=/floppy/datasets/KITTI/object \
                        --epochs=100 --batch_size=6 --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=kitti \
                        --dense_depth --concat_latent_vector \
                        --use_pretrained_weight #--pretrained_weights=./models/vr3d.learning_rate_0.0001.n_xgrids_16.n_ygrids_16.xlim_0.0_70.0.ylim_-25.0_25.0.zlim_-2.5_1.0.max_depth_100.0.vol_size_256x256x16.img_size_512x256.dense_depth_True.concat_latent_vector_True.exp_id_kitti/checkpoint_best.pt
    fi

    if [ "$OPTION" == "ablation" ]; then
        # train for ablation study
        python train.py --dataroot=/floppy/datasets/KITTI/object \
                        --epochs=100 --batch_size=8 --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=kitti_no_latent_concat \
                        --lambda_depth_l2_loss=0 --lambda_depth_smooth_loss=0 \
                        --dense_depth
        
        python train.py --dataroot=/floppy/datasets/KITTI/object \
                        --epochs=100 --batch_size=8 --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=kitti_no_latent_concat_add_l2 \
                        --lambda_depth_smooth_loss=0 \
                        --dense_depth

        python train.py --dataroot=/floppy/datasets/KITTI/object \
                        --epochs=100 --batch_size=8 --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=kitti_no_latent_concat_add_l2_add_smooth \
                        --alpha_depth_smooth_loss=0 
                        --dense_depth

        python train.py --dataroot=/floppy/datasets/KITTI/object \
                        --epochs=100 --batch_size=8 --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=kitti_no_latent_concat_add_l2_add_edge_preserv_smooth \
                        --dense_depth

        python train.py --dataroot=/floppy/datasets/KITTI/object \
                        --epochs=100 --batch_size=8 --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=kitti_latent_concat_add_l2_add_edge_preserv_smooth \
                        --dense_depth --concat_latent_vector --use_pretrained_weight
    fi

elif [ "$MODE" == "test" ]; then
	# test
    python test.py --pc_dir /floppy/datasets/KITTI/raw/2011_09_26_drive_0104_sync/2011_09_26/2011_09_26_drive_0104_sync/velodyne_points/data \
                   --img_dir /floppy/datasets/KITTI/raw/2011_09_26_drive_0104_sync/2011_09_26/2011_09_26_drive_0104_sync/image_02/data \
                   --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --dense_depth --concat_latent_vector --exp_id=kitti

elif [ "$MODE" == "test_and_compare" ]; then
	# test and compare
    python test_and_compare.py --dataroot=/floppy/datasets/KITTI/object \
                               --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --dense_depth --concat_latent_vector --exp_id=kitti

elif [ "$MODE" == "evaluate" ]; then

    if [ "$OPTION" == "none" ]; then
        # evaluate
        python src/eval_kitti.py --pc_dir /floppy/datasets/KITTI/object/testing/velodyne \
                                 --img_dir /floppy/datasets/KITTI/object/testing/image_2 \
                                 --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 \
                                 --dense_depth --concat_latent_vector --eval_object --eval_depth --exp_id=kitti
    fi

    if [ "$OPTION" == "ablation" ]; then
        # evaluate for ablation study
        python src/eval_kitti.py --pc_dir /floppy/datasets/KITTI/object/testing/velodyne \
                                 --img_dir /floppy/datasets/KITTI/object/testing/image_2 \
                                 --epochs=100 --batch_size=8 --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=kitti_no_latent_concat \
                                 --lambda_depth_l2_loss=0 --lambda_depth_smooth_loss=0 \
                                 --dense_depth --eval_depth
        
        python src/eval_kitti.py --pc_dir /floppy/datasets/KITTI/object/testing/velodyne \
                                 --img_dir /floppy/datasets/KITTI/object/testing/image_2 \
                                 --epochs=100 --batch_size=8 --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=kitti_no_latent_concat_add_l2 \
                                 --lambda_depth_smooth_loss=0 \
                                 --dense_depth --eval_depth

        python src/eval_kitti.py --pc_dir /floppy/datasets/KITTI/object/testing/velodyne \
                                 --img_dir /floppy/datasets/KITTI/object/testing/image_2 \
                                 --epochs=100 --batch_size=8 --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=kitti_no_latent_concat_add_l2_add_smooth \
                                 --dense_depth --eval_depth

        python src/eval_kitti.py --pc_dir /floppy/datasets/KITTI/object/testing/velodyne \
                                 --img_dir /floppy/datasets/KITTI/object/testing/image_2 \
                                 --epochs=100 --batch_size=8 --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=kitti_no_latent_concat_add_l2_add_edge_preserv_smooth \
                                 --dense_depth --eval_depth

        python src/eval_kitti.py --pc_dir /floppy/datasets/KITTI/object/testing/velodyne \
                                 --img_dir /floppy/datasets/KITTI/object/testing/image_2 \
                                 --epochs=100 --batch_size=8 --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=kitti_latent_concat_add_l2_add_edge_preserv_smooth \
                                 --dense_depth --eval_depth --concat_latent_vector
    fi

elif [ "$MODE" == "compute_map" ]; then
	# evaluate
    python src/compute_precision_recall.py --dataroot=/floppy/datasets/KITTI/object \
                                           --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --dense_depth --concat_latent_vector --exp_id=kitti  
    python src/compute_map.py --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --dense_depth --concat_latent_vector --exp_id=kitti   
else
    # error
    echo -e "\e[91mMode not provided. Please run this script with one of the following modes..
            \t ./run_experiments train [ablation]
            \t ./run_experiments test
            \t ./run_experiments evaluate [ablation]
            \t ./run_experiments compute_map
            \t ./run_experiments test_and_compare \e[0m"
fi