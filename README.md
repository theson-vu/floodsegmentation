# floodsegmentation

This repository contains PyTorch implementations of U-Net models for semantic segmentation with wavelet, dft and ResNet50 features using the library segmentation_models_pytorch.
The models have been trained and evaluated on **Cloud to Street - Microsoft Flood Dataset** dataset and are composed of Sentinel-1 and Sentinel-2 image chips with corresponding water labels.

## Before starting
* Download data: https://eod-grss-ieee.com/dataset-detail/UXJjWHNsM1AyZGRKMFljLzhJeDN3UT09
* Paths have to be adapted

## Following Encoder configurations exist:
* --image
* --wavelet haar
* --dft
* --deep
* --image --wavelet haar
* --image --dft
* --image --deep

## Structure
* dataset.py: Contains the dataset model for the dataloader. Also select the input data in the get_paths() function here!
* utils.py: Lossfunction, collate function for dataloading
* encoder.py: Encoder architectures and configurations for segmentation_models_pytorch (https://github.com/qubvel-org/segmentation_models.pytorch)
* train_model.py: main trainingloop, hyperparameters and logging

## Quick start
* To run the baseline model
    python train_model.py --image
* more parameters
    *--name <run_name>
    *--batchsize; default=8
    *--epochs; default=20
    *--lr; default=1e-3
    *--seed; default=42
    *--naug; default=2
    *--sen1; when using sen1 data (you have to adapt the filepaths for that)