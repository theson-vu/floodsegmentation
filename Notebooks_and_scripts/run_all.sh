#!/bin/bash

#echo "Running Baseline"
#python train_model.py --name sen2_baseline_final_seed1 --epochs 50 --image --lr 1e-4 --seed 1

#echo "Running Baseline"
#python train_model.py --name sen2_baseline_final_seed2 --epochs 50 --image --lr 1e-4 --seed 2

#echo "Running Baseline"
#python train_model.py --name sen2_baseline_final_seed3 --epochs 50 --image --lr 1e-4 --seed 3

#echo "Running Baseline + DFT 1"
#python train_model.py --name sen1_baseline_dft_final_seed1 --epochs 50 --dft --image --dft --lr 1e-4 --seed 1

#echo "Running Baseline + DFT 2"
#python train_model.py --name sen1_baseline_dft_final_seed2 --epochs 50 --dft --image --dft --lr 1e-4 --seed 2

#echo "Running Baseline + DFT 3"
#python train_model.py --name sen1_baseline_dft_final_seed3 --epochs 50 --dft --image --dft --lr 1e-4 --seed 3

#echo "Running Baseline + wavelet"
#python train_model.py --name sen1_baseline_wavelet2 --epochs 50 --image --wavelet haar --lr 1e-4

echo "Running Baseline + deep 1"
python train_model.py --name sen2_baseline_deep_final_seed1 --epochs 50 --image --deep --lr 1e-4 --seed 1
echo "Running Baseline + deep 2"
python train_model.py --name sen2_baseline_deep_final_seed2 --epochs 50 --image --deep --lr 1e-4 --seed 2
echo "Running Baseline + deep 3"
python train_model.py --name sen2_baseline_deep_final_seed3 --epochs 50 --image --deep --lr 1e-4 --seed 3

echo "Running deep 1"
python train_model.py --name sen2_deep_final_seed1 --epochs 50 --deep --lr 1e-4 --seed 1
echo "Running deep 2"
python train_model.py --name sen2_deep_final_seed2 --epochs 50 --deep --lr 1e-4 --seed 2
echo "Running deep 3"
python train_model.py --name sen2_deep_final_seed3 --epochs 50 --deep --lr 1e-4 --seed 3

#echo "Running DFT"
#python train_model.py --name sen2_dft_final_seed1 --epochs 50 --dft --lr 1e-4 --seed 1
#echo "Running DFT"
#python train_model.py --name sen2_dft_final_seed2 --epochs 50 --dft --lr 1e-4 --seed 2
#echo "Running DFT"
#python train_model.py --name sen2_dft_final_seed3 --epochs 50 --dft --lr 1e-4 --seed 3

#echo "Running wavelet 1"
#python train_model.py --name sen2_wavelet_final_seed1 --epochs 50 --wavelet haar --lr 1e-4 --seed 1

#echo "Running wavelet 2"
#python train_model.py --name sen2_wavelet_final_seed2 --epochs 50 --wavelet haar --lr 1e-4 --seed 2

#echo "Running wavelet 3"
#python train_model.py --name sen2_wavelet_final_seed3 --epochs 50 --wavelet haar --lr 1e-4 --seed 3

#echo "Running baseline_wavelet 1"
#python train_model.py --name sen2_baseline_wavelet_final_seed1 --epochs 50 --image --wavelet haar --lr 1e-4 --seed 1

#echo "Running baseline_wavelet 2"
#python train_model.py --name sen2_baseline_wavelet_final_seed2 --epochs 50 --image --wavelet haar --lr 1e-4 --seed 2

#echo "Running baseline_wavelet 3"
#python train_model.py --name sen2_baseline_wavelet_final_seed3 --epochs 50 --image --wavelet haar --lr 1e-4 --seed 3
