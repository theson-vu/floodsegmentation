import torch
import tifffile
import numpy as np
import cv2
import glob
import os
import time
import albumentations as A
import pywt


class Data():
    def __init__(
            self,
            data_paths,
            mask_paths,
            n_aug,
            transforms=False,
            wavelet="haar"
            ):
        self.data_paths = data_paths
        self.mask_paths = mask_paths
        self.n_aug = n_aug
        self.wavelet = wavelet
        self.transforms = transforms
        self.mean, self.std, self.amp_mean, self.amp_std, self.phase_mean, self.phase_std = self.get_dataset_mean_std()
  

    def get_dataset_mean_std(self):
        # Compute mean while not overloading RAM
        n_images = 0
        mean, M2 = None, None
        amp_mean, amp_M2 = None, None
        phase_mean, phase_M2 = None, None

        for path in self.data_paths:
            if "RFCC.png" in path:
                img = cv2.imread(path) / 255.0
            else:
                img = merge_tif(path)
            img = img.astype(np.float32)  # Ensure the image is in float32 format

            if img.ndim == 2:  # Single-channel image
                img = np.expand_dims(img, axis=-1)

            a, p = [], []
            for channel in img.transpose(2, 0, 1):  # Iterate over channels
                amp, phase = self.compute_amplitude_phase(channel)
                a.append(amp)
                p.append(phase)

            a = np.array(a)
            p = np.array(p)

            n_images += 1

            if n_images == 1:
                mean = np.mean(img, axis=(0, 1))
                M2 = np.zeros_like(mean)
                amp_mean = np.mean(a, axis=(1, 2))
                amp_M2 = np.zeros_like(amp_mean)
                phase_mean = np.mean(p, axis=(1, 2))
                phase_M2 = np.zeros_like(phase_mean)
            else:
                img_mean = np.mean(img, axis=(0, 1))
                delta = img_mean - mean
                mean += delta / n_images
                M2 += delta * (img_mean - mean)

                amp_img_mean = np.mean(a, axis=(1, 2))
                delta_amp = amp_img_mean - amp_mean
                amp_mean += delta_amp / n_images
                amp_M2 += delta_amp * (amp_img_mean - amp_mean)

                phase_img_mean = np.mean(p, axis=(1, 2))
                delta_phase = phase_img_mean - phase_mean
                phase_mean += delta_phase / n_images
                phase_M2 += delta_phase * (phase_img_mean - phase_mean)

        std = np.sqrt(M2 / (n_images - 1))
        amp_std = np.sqrt(amp_M2 / (n_images - 1))
        phase_std = np.sqrt(phase_M2 / (n_images - 1))
        return mean, std, amp_mean, amp_std, phase_mean, phase_std

    def compute_amplitude_phase(self, image_channel):
        # Compute the 2D DFT of the channel
        dft = np.fft.fft2(image_channel)
        dft_shifted = np.fft.fftshift(dft)
        
        # Compute the amplitude spectrum
        amplitude_spectrum = np.abs(dft_shifted)
        
        # Compute the phase spectrum
        phase_spectrum = np.angle(dft_shifted)
        
        return amplitude_spectrum, phase_spectrum


    def apply_augmentations(self, img, mask):
        augmented_images = []
        augmented_masks = []
        replay_params = []

        # Apply augmentations num_augmentations times to the first image
        for _ in range(self.n_aug):
            augmented = self.transforms(image=img, mask=mask)
            augmented_images.append(augmented['image'])
            augmented_masks.append(augmented['mask'])
            replay_params.append(augmented['replay'])
        
        return augmented_images, augmented_masks, replay_params
    
    def normalize(self, image, mean, std):        
        return (image - mean) / std

    def replay_augmentations(self, img, mask, replay_params):
        augmented_images = []
        augmented_masks = []
        for params in replay_params:
            augmented = A.ReplayCompose.replay(params, image=img, mask=mask)
            augmented_images.append(augmented['image'])
            augmented_masks.append(augmented['mask'])
        return augmented_images, augmented_masks

    def wavelet_transform(self, x, wavelet):
        coeffs2 = pywt.dwt2(x, wavelet)
        LL, (LH, HL, HH) = coeffs2
        return np.stack([LL, LH, HL, HH], axis=0)

    def save_precomputed_data(self, output_dir):
        start_time = time.time()
        
        if not os.path.exists(output_dir):
            print("Creating data")        
            os.makedirs(output_dir, exist_ok=True)
            
            for i, path in enumerate(self.data_paths):
                samples = []
                if "RFCC.png" in path:
                    img = cv2.imread(path) / 255.0
                else:
                    img = merge_tif(path)
                img = img.astype(np.float32)  # Ensure the image is in float32 format
                
                if img.ndim == 2:  # Single-channel image
                    img = np.expand_dims(img, axis=-1)

                mask = tifffile.imread(self.mask_paths[i])    
                normalized_img = self.normalize(img, self.mean, self.std)
                replay_params = None
                if self.transforms:
                    if replay_params is None:
                        augmented_samples, augmented_masks, replay_params = self.apply_augmentations(normalized_img, mask)
                    else:
                        augmented_samples, augmented_masks = self.replay_augmentations(normalized_img, mask, replay_params)

                    for augmented_img, augmented_mask in zip(augmented_samples, augmented_masks):
                        a, p = [], []
                        wavelets = []                      
                        for n in range(augmented_img.shape[2]):
                            channel = augmented_img[:, :, n]
                            wavelet = self.wavelet_transform(channel, self.wavelet)
                            amp, phase = self.compute_amplitude_phase(channel)
                            a.append(amp)
                            p.append(phase)
                            wavelets.append(wavelet)

                        augmented_amplitude = self.normalize(np.stack(a, axis=0), self.amp_mean.reshape(self.amp_mean.shape[0], 1, 1), self.amp_std.reshape(self.amp_mean.shape[0], 1, 1))
                        augmented_phase = self.normalize(np.stack(p, axis=0), self.phase_mean.reshape(self.phase_mean.shape[0], 1, 1), self.phase_std.reshape(self.phase_std.shape[0], 1, 1))

                        #augmented_mean = np.mean(augmented_img, axis=2)
                        #augmented_wavelet = self.wavelet_transform(augmented_mean, self.wavelet)
                        augmented_wavelet = np.concatenate(wavelets, axis=0)

                        sample = {
                            "image": augmented_img.transpose(2, 0, 1),
                            "amplitude": augmented_amplitude,
                            "phase": augmented_phase,
                            "wavelet": augmented_wavelet,
                            "mask": augmented_mask
                        }
                        samples.append(sample)
                else:
                    a, p = [], []
                    wavelets = []
                    #img_mean = np.mean(normalized_img, axis=2)
                    #wavelets_ = self.wavelet_transform(img_mean, self.wavelet)
                    for n in range(img.shape[2]):
                        channel = img[:, :, n]
                        wavelet = self.wavelet_transform(channel, self.wavelet)
                        amp, phase = self.compute_amplitude_phase(channel)
                        a.append(amp)
                        p.append(phase)
                        wavelets.append(wavelet)
                    normalized_amplitude = self.normalize(np.stack(a, axis=0), self.amp_mean.reshape(self.amp_mean.shape[0], 1, 1), self.amp_std.reshape(self.amp_mean.shape[0], 1, 1))
                    normalized_phase = self.normalize(np.stack(p, axis=0), self.phase_mean.reshape(self.phase_mean.shape[0], 1, 1), self.phase_std.reshape(self.phase_std.shape[0], 1, 1))
                    wavelets_ = np.concatenate(wavelets, axis=0)
                    sample = {
                        "image": normalized_img.transpose(2, 0, 1),
                        "amplitude": normalized_amplitude,
                        "phase": normalized_phase,
                        "wavelet": wavelets_,
                        "mask": mask
                    }
                    samples.append(sample)
                output_path = f"{output_dir}/{i}.npz"
                np.savez(output_path, samples)
            print(f"Finished creating data {time.time() - start_time}")



class S2Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            img_paths,
            mask_paths,
            data_path,
            transforms=None,
            seed=1337,
            num_augmentations=4,
            dft_flag=True,
            wavelet="haar",
            normalize=[]            
    ):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transforms = transforms
        self.dft_flag = dft_flag
        self.seed = seed
        self.data_path = data_path
        self.num_augmentations = num_augmentations
        self.wavelet = wavelet
        self.normalize = normalize
        if self.normalize:
            self.mean, self.std, self.amp_mean, self.amp_std, self.phase_mean, self.phase_std = normalize
        else:
            self.mean, self.std, self.amp_mean, self.amp_std, self.phase_mean, self.phase_std = None, None, None, None, None, None

        if not os.path.exists(data_path):
            self.data_classes = self.get_data_classes()
        


    def get_data_classes(self):
        "Creates a list of Classes containing information about the inputdata"
        data_classes = []
        for data in self.img_paths:
            data_class = Data(data, self.mask_paths, wavelet=self.wavelet, transforms=self.transforms, n_aug=self.num_augmentations)
            if self.normalize:
                data_class.mean, data_class.std, data_class.amp_mean, data_class.amp_std, data_class.phase_mean, data_class.phase_std = self.mean, self.std, self.amp_mean, self.amp_std, self.phase_mean, self.phase_std
            else:
                self.mean, self.std, self.amp_mean, self.amp_std, self.phase_mean, self.phase_std = data_class.mean, data_class.std, data_class.amp_mean, data_class.amp_std, data_class.phase_mean, data_class.phase_std
            data_class.save_precomputed_data(self.data_path)
            data_classes.append(data_class)        
        return data_classes


    def __getitem__(self, idx):
        # Load precomputed data for the specific index
        data = np.load(f"{self.data_path}/{idx}.npz", allow_pickle=True)
        return data["arr_0"]


    def __len__(self):
        return len(self.mask_paths)
    
            
def mask_to_img(label, color_dict):
    """Recodes a (H,W) mask to a (H,W,3) RGB image according to color_dict"""
    mutually_exclusive = np.zeros(label.shape + (3,), dtype=np.uint8)
    for key in range(1, len(color_dict.keys()) + 1):
        mutually_exclusive[label == key] = color_dict[key]
    return mutually_exclusive


def merge_tif(paths):
    img = []
    for tif in paths:
        img.append(tifffile.imread(tif))
    return np.stack(img, axis=-1)


def get_paths(fraction=1, bands=["B2.tif", "B3.tif", "B4.tif",  "B8.tif", "B10.tif", "B12.tif"], seed=1337):
    """
    Returns:
        label_paths: list of all labelpaths
        bands_paths: list of lists containing paths to databands [[Bandpaths1], .., [BandpathsN]]
    """
    label_paths = sorted(glob.glob("Data/ms-dataset-chips/*/s2/*/LabelWater.tif"))
    filtered_paths = []
    # Only select labels, where there is more than 50% of information
    for png in label_paths:
        path = png.replace("LabelWater.tif", bands[0])
        flat_img = np.ravel(cv2.imread(path))
        #flat_img = np.ravel(tifffile.imread(path))
        if np.sum(flat_img == 0) / len(flat_img) <= 0.5:
            filtered_paths.append(png)

    label_paths = filtered_paths[:int(len(filtered_paths)*fraction)]

    np.random.seed(seed)
    np.random.shuffle(label_paths)

    bands_paths = []
    for band in bands:
        if ".tif" in band:
            continue
        band_path_list = []
        for path in label_paths:
            band_path_list.append(path.replace("LabelWater.tif", f"{band}"))
        bands_paths.append(band_path_list)

    tif = []
    for path in label_paths:
        p = []
        for band in bands:
            if band.endswith(".png"):
                continue
            p.append(path.replace("LabelWater.tif", f"{band}"))
        tif.append(p)
    if len(tif) != 1:
        bands_paths.append(tif)
    return label_paths, bands_paths


def create_splits(train_percentage=0.6, val_percentage=0.2, test_percentage=0.2, seed=1337):
    label_paths, img_paths = get_paths(fraction=1, seed=seed)  # Use bands=["RFCC.png"] for sen1
    img_paths = [img_paths[0]]  # Comment out if using .tif

    img_paths_train = [row[: int(train_percentage * len(label_paths))] for row in img_paths]
    img_paths_val = [row[int(train_percentage * len(label_paths)): int((train_percentage + val_percentage) * len(label_paths))] for row in img_paths]
    img_paths_test = [row[int((train_percentage + val_percentage) * len(label_paths)):] for row in img_paths]

    label_paths_train = label_paths[: int(train_percentage * len(label_paths))]
    label_paths_val = label_paths[int(train_percentage * len(label_paths)): int((train_percentage + val_percentage) * len(label_paths))]
    label_paths_test = label_paths[int((train_percentage + val_percentage) * len(label_paths)):]

    print(f"Train has {len(img_paths_train[0])} images and {len(label_paths_train)} labels")
    print(f"Val   has {len(img_paths_val[0])} images and {len(label_paths_val)} labels")
    print(f"Test  has {len(img_paths_test[0])} images and {len(label_paths_test)} labels")
    return img_paths_train, img_paths_val, img_paths_test, label_paths_train, label_paths_val, label_paths_test