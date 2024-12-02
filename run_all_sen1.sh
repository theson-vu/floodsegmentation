#!/bin/bash

echo "Running Baseline Sen1"
python train_model.py --name sen1_baseline_final_seed1 --epochs 50 --sen1 --image --lr 1e-4 --seed 1
echo "Running Baseline Sen1"
python train_model.py --name sen1_baseline_final_seed2 --epochs 50 --sen1 --image --lr 1e-4 --seed 2
echo "Running Baseline Sen1"
python train_model.py --name sen1_baseline_final_seed3 --epochs 50 --sen1 --image --lr 1e-4 --seed 3

echo "Running Baseline + DFT 1 Sen1"
python train_model.py --name sen1_baseline_dft_final_seed1 --epochs 50 --sen1 --dft --image --dft --lr 1e-4 --seed 1
echo "Running Baseline + DFT 2 Sen1"
python train_model.py --name sen1_baseline_dft_final_seed2 --epochs 50 --sen1 --dft --image --dft --lr 1e-4 --seed 2
echo "Running Baseline + DFT 3 Sen1"
python train_model.py --name sen1_baseline_dft_final_seed3 --epochs 50 --sen1 --dft --image --dft --lr 1e-4 --seed 3


echo "Running DFT Sen1"
python train_model.py --name sen1_dft_final_seed1 --epochs 50 --sen1 --dft --lr 1e-4 --seed 1
echo "Running DFT Sen1"
python train_model.py --name sen1_dft_final_seed2 --epochs 50 --sen1 --dft --lr 1e-4 --seed 2
echo "Running DFT Sen1"
python train_model.py --name sen1_dft_final_seed3 --epochs 50 --sen1 --dft --lr 1e-4 --seed 3

echo "Running wavelet 1 Sen1"
python train_model.py --name sen1_wavelet_final_seed1 --epochs 50 --sen1 --wavelet haar --lr 1e-4 --seed 1
echo "Running wavelet 2 Sen1"
python train_model.py --name sen1_wavelet_final_seed2 --epochs 50 --sen1 --wavelet haar --lr 1e-4 --seed 2
echo "Running wavelet 3 Sen1"
python train_model.py --name sen1_wavelet_final_seed3 --epochs 50 --sen1 --wavelet haar --lr 1e-4 --seed 3

echo "Running baseline_wavelet 1 Sen1"
python train_model.py --name sen1_baseline_wavelet_final_seed1 --epochs 50 --sen1 --image --wavelet haar --lr 1e-4 --seed 1
echo "Running baseline_wavelet 2 Sen1"
python train_model.py --name sen1_baseline_wavelet_final_seed2 --epochs 50 --sen1 --image --wavelet haar --lr 1e-4 --seed 2
echo "Running baseline_wavelet 3 Sen1"
python train_model.py --name sen1_baseline_wavelet_final_seed3 --epochs 50 --sen1 --image --wavelet haar --lr 1e-4 --seed 3
