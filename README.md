# VR3Dense: 3D Object Detection and Dense Depth Reconstruction from Voxel-Representation

## Dependencies 

This project has been tested with `PyTorch=1.7.1` and `cuda-11.0`.  

## Environment Setup  

Install conda environment as: ```conda env create -f conda_env.yml```.  

## Training and Testing VR3Dense  

The network can be trained as: ```./run_experiments train```.  
The network can be test as: ```./run_experiments test```.  

## Multi-Object Tracking  

Multi-object tracking can be selectively turned on or off. This repo uses **[AB3DMOT](https://github.com/xinshuoweng/AB3DMOT)** for tracking.

## Results  


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

 - Open the file ```kitti_devkit_object/cpp/evaluate_object.cpp``` and update the number of validation samples by modifying the line ```const int32_t N_TESTIMAGES = 1000;```.   
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
  ./evaluate_obj exp
 ```

 - This performs the evaluation and generates report with plots. 

 - A python script is also provided at ```kitti_devkit_object/cpp/parser.py``` to parse the report and summarize it.  

 ```
 cd ..
 python parser exp
 ```
