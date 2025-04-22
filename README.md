# FibEmNet: Dual-Pathology classification and detection using CT images

This repositary outlines the end-to-end classification and detection framework of pulmonary fibrosis and emphysema. The objective of the design is to tackle the data scarity of CPFE(Combined Pulmonary Fibrosis and Emphysema) by training the model with available data of fibrosis and emphysema efficiently to predict the patterns. The pulmonary fibrosis data taken from [Kaggle Fibrosis Dataset](https://www.kaggle.com/datasets/icmicm/pulmonaryfibrosis-dataset-final/data) and emphysema from [official Emphysema dataset page](https://lauge-soerensen.github.io/emphysema-database/) which collectively consists of 2D CT images from healthy and affected patients. 
The architecture consists of combining the data, feature extraction and learning, classification model: 
[Untitled Diagram drawio-21 drawio](https://github.com/user-attachments/assets/246f897f-200b-47db-b279-ae2862fe0446)



## Requirements 
The overall framework setup and trained on Single-GPU - NVIDIA A100. To setup and work with the model: Create conda environment and install the dependencies. If you are facing computing resource shortage, please tune up the depth of model(models/MedViT.py) and switch to T4 GPU.

### Create Environment

### Install Dependencies
requirements.txt

## Workflow & Model Setup
Models used in this work are built using MedViT @ and Customized CSwin @ 

### Dataset 
To download the datasets, please refer the dataset folder and make up the folders
Pulmonary_data
  -- Normal
    --nc1.jpg
  -- Fibrosis
    --1.jpg
Emphysema_data
   -- Patches
   -- Slices
   -- Patch_label.csv
   -- Slice_label.csv

### Training

### Inference
