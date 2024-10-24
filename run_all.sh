#!/bin/bash

echo "Running Baseline"
python train_model.py --name baseline2 --epochs 50 --image --lr 1e-4

echo "Running Baseline + DFT"
python train_model.py --name baseline_dft2 --epochs 50 --dft --image --dft --lr 1e-4

echo "Running Baseline + wavelet"
python train_model.py --name baseline_wavelet2 --epochs 50 --image --wavelet haar --lr 1e-4

echo "Running Baseline + deep"
python train_model.py --name baseline_deep2 --epochs 50 --image --deep --lr 1e-4

echo "Running DFT"
python train_model.py --name dft2 --epochs 50 --dft --lr 1e-4

echo "Running deep"
python train_model.py --name deep2 --epochs 50 --deep --lr 1e-4

echo "Running wavelet"
python train_model.py --name wavelet2 --epochs 50 --wavelet haar --lr 1e-4

shutdown /f