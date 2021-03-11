# VR3Dense: 3D Object Detection and Dense Depth Reconstruction from Voxel-Representation

VR3Dense jointly trains for 3D object detection as well as semi-supervised dense depth reconstruction. Object detection uses 3D convolutions over voxelized point-cloud to obtain 3D bounding-boxes, while dense depth reconstruction network uses an *hourglass* architecture with skip connections. The complete pipeline has been trained on KITTI object training dataset with 90% train and 10% validation split. VR3Dense supports detection and classification of the following classes: ```Car, Cyclist, Pedestrian```. VR3Dense runs at about **139.89fps** on a PC with i9 10850K processor with single NVIDIA RTX 3090 GPU.  

## Our Approach

![](media/VR3Dense_Approach.png)

## Model Predictions 
![](media/demo.gif)  
*Figure 1 - VR3Dense tested on KITTI raw dataset | Date:2011-09-26 | Sequence: 0009*  

![](media/demo_scene104.gif)  
*Figure 2 - VR3Dense tested on KITTI raw dataset | Date:2011-09-26 | Sequence: 0104*  

## Dependencies 

This project has been tested with `PyTorch=1.7.1` and `cuda-11.0`.  

## Environment Setup  

Install conda environment as: ```conda env create -f conda_env.yml```.  

## Training, Testing, and Evaluation of VR3Dense  

### 1. Training on KITTI Object Dataset

Please download *left*, *right*, *velodyne* data, and *labels* from here: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d and extract them. 
The network can be trained as: 
```
python train.py --dataroot={KITTI_OBJECT_TRAINING_DATAROOT} \
                --epochs=100 --batch_size=8 --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=kitti \
                --dense_depth --concat_latent_vector 
```

Or alternatively, you can run the provided bash script as: ```./run_experiments train```.  

### 2. Testing the pre-trained model on KITTI dataset

You need a set of *left image* files and the corresponding *velodyne point-cloud* files during testing. You can download KITTI raw, or object dataset for testing. Set up the paths correctly in *test.py* and then run:

```
python test.py --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=kitti --dense_depth --concat_latent_vector 
```

This will download the pre-trained model if you do not have it locally and then run inference on it. You can choose to use a multi-object tracker by modifying the *TRACKING* parameter within *test.py*. This work uses **[AB3DMOT](https://github.com/xinshuoweng/AB3DMOT)** for multi-object tracking.

The network can also be test by simply running: ```./run_experiments test```.  

A demo of the inference pipeline is shown in ```src/misc/demo.ipynb```.
### 3. Evaluation of the model on KITTI dataset

For evaluation of the model, you need a set of *left image* files and the corresponding *velodyne point-cloud* files. Set up the paths correctly in *src/eval_kitti.py* and then run: 

```
python src/eval_kitti.py --dataroot={KITTI_OBJECT_TESTING_DATAROOT} \
                         --learning_rate=0.0001 --n_xgrids=16 --n_ygrids=16 --exp_id=kitti --dense_depth --concat_latent_vector 
```

This will generate a set of *.txt* files for object detection, and compute *depth prediction* metrics for each file - which will be summarized at the end. Once you have obtained a set of *.txt* files, you can either use the *kitti object evaluation kit* to compute performance, or alternatively *zip* them and submit to KITTI evaluation server. Please follow [this](#kitti-evaluation) section for KITTI evaluation locally.

## Quantitative Results  

### 3D Object Detection

3D object detection results on KITTI Object dataset (*eval* split) is summarized in the table below:

```
Easy    Mod.    Hard
2.01    2.05    2.41
```

### Dense Depth Reconstruction

Dense Depth Map prediction is evaluated based on the pixels for which we have lidar point projection available. Quantitative results on KITTI object *testing* dataset (7518 images) are given below:  

```
| ========================================================================== |
| abs_rel  | sq_rel   | rmse     | rmse_log | a1       | a2       | a3       |
| ========================================================================== |
| 0.219240 | 3.859928 | 7.164907 | 0.302273 | 0.760768 | 0.888690 | 0.944800 |
| ========================================================================== |
```

## KITTI Evaluation

To write predicted labels to files in the KITTI format: ```./run_experiments evaluate```

Evaluations can be performed using KITTI object evaluation kit, included in this repo (kitti_devkit_object). This kit expects ground-truth in KITTI format, one *txt* file per frame, numbered sequentially starting from *000000.txt*. If using a KITTI dataset split, ```src/kitti_split.py``` can be used to randomly split the training set into train/val split and store the validation split by naming them sequentially. 

Evaluation kit has the following prerequisites:  
```
boost   (sudo apt-get install libboost-all-dev)
pdfcrop (sudo apt-get install texlive-extra-utils)
gnuplot (sudo apt-get install gnuplot)
```  

Install the prerequisites and follow the steps below to perform evaluations.

 - Open the file ```kitti_devkit_object/cpp/evaluate_object.cpp``` and update the number of validation samples by modifying the line ```const int32_t N_TESTIMAGES = 7518;```.   
 - Build binary.  

   ```
   cd kitti_devkit_object/cpp
   mkdir build
   cd build
   cmake ..
   make
   ```

 - Create a folder for ground truth files at ```build/data/object/``` and create a symlink for ground-truth folder.  

  ```
   mkdir -p data/object  
   cd data/object
   ln -s /path/to/groundtruth/object/label_2/ label_2
   cd ../../
  ```

 - Create a folder for current experiment at ```results/exp/```, where ```exp``` is the experiment id. Create a symlink for prediction folder within the experiment folder with name ```data```.  

  ```
   mkdir -p results/exp 
   cd results/exp
   ln -s /path/to/prediction/label/ data
   cd ../../
  ```

 - Run the evaluation kit with this experiment ID.  

 ```
  ./evaluate_object exp
 ```

 - This performs the evaluation and generates report with plots. 

 - A python script is also provided at ```kitti_devkit_object/cpp/parser.py``` to parse the report and summarize it.  

 ```
 cd ..
 python parser exp
 ```
