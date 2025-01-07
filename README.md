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
* **dataset.py**: Contains the dataset model for the dataloader. It also selects the input data in the `get_paths()` function here.
* **utils.py**: Contains the loss function and collate function for dataloading.
* **encoder.py**: Encoder architectures and configurations for [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch).
* **train_model.py**: Main training loop, hyperparameters, and logging.

## Quick Start
* To run the baseline model:
    ```bash
    python train_model.py --image
    ```

* More parameters:
    * `--name <run_name>`: Specify the run name.
    * `--batchsize`: Default is 8.
    * `--epochs`: Default is 20.
    * `--lr`: Default is 1e-3.
    * `--seed`: Default is 42.
    * `--naug`: Default is 2.
    * `--sen1`: Use when working with `sen1` data (you need to adapt the file paths for that).
