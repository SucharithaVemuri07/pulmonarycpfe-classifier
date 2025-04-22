# FibEmNet: Dual-Pathology classification and detection using CT images

This repositary outlines the end-to-end classification and detection framework of pulmonary fibrosis and emphysema. The objective of the design is to tackle the data scarity of CPFE(Combined Pulmonary Fibrosis and Emphysema) by training the model with available data of fibrosis and emphysema efficiently to predict the patterns. The pulmonary fibrosis data taken from [Kaggle Fibrosis Dataset](https://www.kaggle.com/datasets/icmicm/pulmonaryfibrosis-dataset-final/data) and emphysema from [official Emphysema dataset page](https://lauge-soerensen.github.io/emphysema-database/) which collectively consists of 2D CT images from healthy and affected patients. 
The architecture consists of combining the data, feature extraction and learning, classification model: 
![block-diagram](https://github.com/user-attachments/assets/246f897f-200b-47db-b279-ae2862fe0446)


## Requirements 
The overall framework setup and trained on Single-GPU - NVIDIA A100. To setup and work with the model: Create conda environment and install the dependencies. If you are facing computing resource shortage, please tune up the depth of model(models/MedViT.py) and switch to NVIDIA T4 GPU.

### Create Environment
```bash
conda create -n <env-name> python=3.11
conda activate <env-name>
```

### Install Dependencies
```bash
pip install requirements.txt
```

## Workflow & Model Setup
Models used in this work are built using [MedViT](https://github.com/Omid-Nejati/MedViT) and Customized [CSwin](https://github.com/microsoft/CSWin-Transformer). 

### Dataset 
To download the datasets, please refer to the dataset folder and create the following local folder structure:

```
Pulmonary_data/
├── Normal/
│   └── nc1.jpg
└── Fibrosis/
    └── 1.jpg

Emphysema_data/
├── Patches/
├── Slices/
├── Patch_label.csv
└── Slice_label.csv
```

### Training
To train the model, clone the repo and run the following command. The script loads both fibrosis and emphysema datasets, builds the FusionMedViT model (CSWin + MedViT), and trains it using confidence-based late fusion.
```bash
python train.py
```

### Inference
Evaluate the model by loading the best model and validate the predictions
```bash
python predict.py
```

### Results & Insights
The model shows strong performance on the dominant classes (NORMAL, FIBROSIS). Despite class imbalance, NT and CLE of emphysema predictions improved significantly with meaningful recall. MCC of 0.7031 confirms that even with imbalanced data, the model maintains overall predictive strength.
```
Metric                         Value
-----------------------------  -------
Overall Accuracy               87.35 %
Macro F1‑Score                 64.0 %
Macro AUC                      96.49 %
Matthews Corr. Coefficient     0.7031
```

## Acknowledgments
- [CPFE paper](https://www.medrxiv.org/content/10.1101/2025.01.20.25320811v3) - Implementation of CPFE using DL 
- [Fibrosis](https://www.nature.com/articles/s41586-020-2938-9) - Pulmonary Fibrosis - Mechanism to Medicine
- [Emphysema](https://www.atsjournals.org/doi/abs/10.1513/pats.200708-126et) - Details of Emphysema

Thanks to the open-source community and CPFE researchers! 

If you really liked and planning to use the idea in your work, please cite this repository: 
@misc{fibemnet2025,
  title        = {FibEmNet: Dual-Pathology classification and detection using CT images},
  author       = {Sucharitha Vemuri},
  year         = {2025},
  howpublished = {\url{https://github.com/SucharithaVemuri07/pulmonarycpfe-classifier}},
  note         = {Work presented at the University of Central Florida}
}

Any questions or suggestions, feel free to reach out!
