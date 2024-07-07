import torch
import tifffile
import numpy as np
import cv2
import glob
import albumentations as A


class Data():
    def __init__(
            self,
            data_paths,
            ):
        self.data_paths = data_paths
        self.name = self.data_paths[0].split("\\")[-1]
        self.num_channels = self.get_num_channels()
        self.mean, self.std, self.amp_mean, self.amp_std, self.phase_mean, self.phase_std = self.get_dataset_mean_std()


    def get_num_channels(self):
        return 1 if self.data_paths[0].endswith(".tif") else 3
    

    def get_dataset_mean_std(self):
        n_images = 0
        mean, M2 = None, None
        amp_mean, amp_M2 = None, None
        phase_mean, phase_M2 = None, None

        for path in self.data_paths:
            if path.endswith(".tif"):
                img = tifffile.imread(path)
            elif path.endswith(".png"):
                img = cv2.imread(path) / 255.0
            else:
                continue  # Skip unsupported file types

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




class S2Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            img_paths,
            mask_paths,
            transforms=None,
            seed=1337,
            num_augmentations=4,
            dft_flag=True,
    ):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transforms = transforms
        self.dft_flag = dft_flag
        self.seed = seed
        self.num_augmentations = num_augmentations
        self.dim = 0
        self.data_classes = self.get_data_classes()


    def get_data_classes(self):
        "Creates a list of Classes containing information about the inputdata"
        data_classes = []
        for data_class in self.img_paths:
            data_classes.append(Data(data_class))
        for i in data_classes:
            if self.dft_flag:
                self.dim += (i.num_channels * 3)
            else:
                self.dim += i.num_channels
        return data_classes
    

    def apply_augmentations(self, img, mask):
        augmented_images = []
        augmented_masks = []
        replay_params = []

        # Apply augmentations num_augmentations times to the first image
        for _ in range(self.num_augmentations):
            augmented = self.transforms(image=img, mask=mask)
            augmented_images.append(augmented['image'])
            augmented_masks.append(augmented['mask'])
            replay_params.append(augmented['replay'])
        
        return augmented_images, augmented_masks, replay_params
    

    def replay_augmentations(self, img, mask, replay_params):
        augmented_images = []
        augmented_masks = []
        for params in replay_params:
            augmented = A.ReplayCompose.replay(params, image=img, mask=mask)
            augmented_images.append(augmented['image'])
            augmented_masks.append(augmented['mask'])
        return augmented_images, augmented_masks


    def normalize(self, image, mean, std):
        return (image - mean) / std


    def __getitem__(self, idx):
        replay_params = None
        mask = None
        result = None
        sample = []
        masks = []
        for i, c in enumerate(self.data_classes):
            mask = tifffile.imread(self.mask_paths[idx])
            if c.data_paths[idx].endswith(".tif"):
                img = tifffile.imread(c.data_paths[idx])
            elif c.data_paths[idx].endswith(".png"):
                img = cv2.imread(c.data_paths[idx]) / 255.0
            if img.ndim == 2:  # Single-channel image
                img = np.expand_dims(img, axis=-1)
            normalized_img = self.normalize(img, c.mean, c.std)
            if self.transforms:
                if replay_params == None:
                    augmented_samples, mask, replay_params = self.apply_augmentations(normalized_img, mask)
                else:
                    # Replay augmentations for subsequent images
                    augmented_samples, mask = self.replay_augmentations(normalized_img, mask, replay_params)

                if self.dft_flag:
                    samples = []
                    for augmented_img in augmented_samples:
                        #augmented_img = augmented_img.transpose(2, 0, 1)  # Change to (C, H, W) format
                        a, p = [], []
                        for channel in augmented_img:  # Iterate over channels
                            amp, phase = c.compute_amplitude_phase(channel)
                            a.append(amp)
                            p.append(phase)
                        # Stack amplitude and phase along the channel dimension
                        augmented_amplitude = self.normalize(np.stack(a, axis=0), c.amp_mean, c.amp_std)
                        augmented_phase = self.normalize(np.stack(p, axis=0), c.phase_mean, c.phase_std)
                        # Combine amplitude and phase into a single array and change to C,H,W
                        samples.append(np.concatenate([augmented_img, augmented_amplitude, augmented_phase], axis=-1))
                    sample.append(samples)
                    masks = mask
                    continue
                sample.append(augmented_samples)
                masks = mask
                continue
            if self.dft_flag:
                a, p = [], []
                for channel in normalized_img:  # Iterate over channels
                    amp, phase = c.compute_amplitude_phase(channel)
                    a.append(amp)
                    p.append(phase)
                # Stack amplitude and phase along the channel dimension
                normalized_amplitude = self.normalize(np.stack(a, axis=0), c.amp_mean, c.amp_std)
                normalized_phase = self.normalize(np.stack(p, axis=0), c.phase_mean, c.phase_std)
                sample.append(np.concatenate([normalized_img, normalized_amplitude, normalized_phase], axis=-1))
                continue
            sample.append(np.array(normalized_img))
            masks.append(mask)
            continue
        if self.transforms:
            arr_x = [np.concatenate(arr, axis=-1) for arr in zip(*sample)]
            result = [{"image": arr_x[i].transpose(2, 0, 1), "mask": masks[i]} for i in range(len(arr_x))]
        else:
            arr_x = np.concatenate(sample, axis=-1)
            result = [{"image": arr_x.transpose(2, 0, 1), "mask": mask}]
        return result

    def __len__(self):
        return len(self.mask_paths)
    


class Sentinel2_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_paths,
        mask_paths,
        transforms=None,
        seed=1337,
        num_augmentations=4,
        dft_flag=True,
    ):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transforms = transforms
        self.dft_flag = dft_flag
        self.seed = seed
        self.num_augmentations = num_augmentations
        self.mean, self.std, self.amp_mean, self.amp_std, self.phase_mean, self.phase_std = self.get_dataset_mean_std()
        self.dim = self.get_dim()


    def get_dim(self):
        if self.dft_flag:
            return len(np.ravel(self.mean)) + len(np.ravel(self.amp_mean)) + len(np.ravel(self.phase_mean))
        else:
            return len(np.ravel(self.mean))

    def get_dataset_mean_std(self):
        """
        Computes the channelwise mean and std of the dataset incrementally
        """
        n_images = 0
        mean, M2 = None, None
        amp_mean, amp_M2 = None, None
        phase_mean, phase_M2 = None, None
        means, stds, amp_means, amp_stds, phase_means, phase_stds = [], [], [], [], [], []

        for path_list in self.img_paths:
            for path in path_list:
                if path.endswith(".tif"):
                    img = tifffile.imread(path)
                elif path.endswith(".png"):
                    img = cv2.imread(path) / 255.0
                else:
                    continue  # Skip unsupported file types

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

            if n_images > 1:
                std = np.sqrt(M2 / (n_images - 1))
                amp_std = np.sqrt(amp_M2 / (n_images - 1))
                phase_std = np.sqrt(phase_M2 / (n_images - 1))
            else:
                std, amp_std, phase_std = [np.zeros_like(mean)] * 3
            means.append(mean)
            stds.append(std)
            amp_means.append(amp_mean)
            amp_stds.append(amp_std)
            phase_means.append(phase_mean)
            phase_stds.append(phase_std)
        print("Finished computing means and stds")
        print(means, stds, amp_means, amp_stds, phase_means, phase_stds)
        return means, stds, amp_means, amp_stds, phase_means, phase_stds
    
    def get_pwater_mean_std(self):
        """
        Computes the mean and std of pwater data
        """
        pwater_paths = [path.replace("LabelWater.tif", f"LabelWater_jrc-gsw-occurrence.tif") for path in self.mask_paths]
        images = []
        for path in pwater_paths:
            img = tifffile.imread(path)
            images.append(img / 100.0)
        return np.mean(images), np.std(images)

    def normalize_data(self, image, index, epsilon=1e-6):
        mean = self.mean[index]
        std = self.std[index]
        normalized_image = (image - mean) / (std)
        return normalized_image
    
    def normalize_amp_phase(self, amp, phase, index, epsilon=1e-6):
        amp_mean = self.amp_mean[index]
        amp_std = self.amp_std[index]

        phase_mean = self.phase_mean[index]
        phase_std = self.phase_std[index]

        normalized_amp = (amp - amp_mean) / (amp_std)
        normalized_phase = (phase - phase_mean) / (phase_std)
        return normalized_amp, normalized_phase
    
    def normalize_pwater(self, image, epsilon=1e-6):
        return (image - self.pwater_mean) / (self.pwater_std)
    
    def compute_amplitude_phase(self, image_channel):
        # Compute the 2D DFT of the channel
        dft = np.fft.fft2(image_channel)
        dft_shifted = np.fft.fftshift(dft)
        
        # Compute the amplitude spectrum
        amplitude_spectrum = np.abs(dft_shifted)
        
        # Compute the phase spectrum
        phase_spectrum = np.angle(dft_shifted)
        
        return amplitude_spectrum, phase_spectrum
    
    def __getitem__(self, idx):
        # Load in image
        arr_x = []
        for i, path_list in enumerate(self.img_paths):
            if path_list[idx].endswith(".tif"):
                img = tifffile.imread(path_list[idx])
            elif path_list[idx].endswith(".png"):
                img = cv2.imread(path_list[idx]) / 255.0
            img_norm = self.normalize_data(img, i)
            arr_x.append(img_norm)
        arr_x = np.concatenate(arr_x, axis=-1)
        #pwater = tifffile.imread(self.mask_paths[idx].replace("LabelWater.tif", f"LabelWater_jrc-gsw-occurrence.tif"))
        #pwater = self.normalize_pwater(pwater / 100.0)
        #pwater = np.expand_dims(pwater, axis=-1)
        #arr_x = np.concatenate((arr_x, pwater), axis=-1)

        sample = {"image": arr_x}
        # Load in label mask
        sample["mask"] = tifffile.imread(self.mask_paths[idx])

        # Apply Data Augmentation
        if self.transforms:
            augmented_samples = []
            for _ in range(self.num_augmentations):
                augmented_sample = self.transforms(image=sample["image"], mask=sample["mask"])
                augmented_samples.append(augmented_sample)
                if augmented_sample["image"].shape[-1] < 20:
                    augmented_sample["image"] = augmented_sample["image"].transpose((2, 0, 1))
            if self.dft_flag:
                augmented_samples_with_spectra = []
                for augmented_sample in augmented_samples:
                    # Extract image and compute amplitude and phase
                    img = augmented_sample["image"].transpose((1, 2, 0))  # Convert back to HWC format
                    amplitude_channels = []
                    phase_channels = []
                    for c in img:
                        amplitude, phase = self.compute_amplitude_phase(c)
                        amplitude_channels.append(amplitude)
                        phase_channels.append(phase)

                    # Normalize amplitude and phase spectra
                    amplitude_channels = np.stack(amplitude_channels, axis=-1).transpose((2, 0, 1))
                    phase_channels = np.stack(phase_channels, axis=-1).transpose((2, 0, 1))
                    amplitude_channels, phase_channels = self.normalize_amp_phase(amplitude_channels, phase_channels, )

                    # Append amplitude and phase to the augmented image
                    augmented_image_with_spectra = np.concatenate([img, amplitude_channels, phase_channels], axis=-1)
                    augmented_sample["image"] = augmented_image_with_spectra.transpose((2, 0, 1))  # Convert back to CHW format
                    augmented_samples_with_spectra.append(augmented_sample)
                return augmented_samples_with_spectra
            else:
                return augmented_samples
        
        # Compute amplitude and phase for each channel in the original image
        img = arr_x
        if self.dft_flag:
            amplitude_channels = []
            phase_channels = []
            for i, c in enumerate(img):
                amplitude, phase = self.normalize_amp_phase(*self.compute_amplitude_phase(c), i)
                amplitude_channels.append(amplitude)
                phase_channels.append(phase)

            # Normalize amplitude and phase spectra
            amplitude_channels = np.stack(amplitude_channels, axis=-1).transpose((2, 0, 1))
            phase_channels = np.stack(phase_channels, axis=-1).transpose((2, 0, 1))

            # Append amplitude and phase to the original image
            arr_x_with_spectra = np.concatenate([arr_x, amplitude_channels, phase_channels], axis=-1)
            sample["image"] = arr_x_with_spectra.transpose((2, 0, 1))  # Convert to CHW format
        if sample["image"].shape[-1] < 20:
            sample["image"] = sample["image"].transpose((2, 0, 1))
        return sample

    def __len__(self):
        return len(self.mask_paths)

            
def mask_to_img(label, color_dict):
    """Recodes a (H,W) mask to a (H,W,3) RGB image according to color_dict"""
    mutually_exclusive = np.zeros(label.shape + (3,), dtype=np.uint8)
    for key in range(1, len(color_dict.keys()) + 1):
        mutually_exclusive[label == key] = color_dict[key]
    return mutually_exclusive



def get_paths(fraction=1, bands=["B3.tif", "B8.tif", "B12.tif"], seed=1337):
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
        if np.sum(flat_img == 0) / len(flat_img) <= 0.5:
            filtered_paths.append(png)

    label_paths = filtered_paths[:int(len(filtered_paths)*fraction)]

    np.random.seed(seed)
    np.random.shuffle(label_paths)

    bands_paths = []
    for band in bands:
        band_path_list = []
        for path in label_paths:
            band_path_list.append(path.replace("LabelWater.tif", f"{band}"))
        bands_paths.append(band_path_list)
    return label_paths, bands_paths


def create_splits(train_percentage=0.7, val_percentage=0.15, test_percentage=0.15, seed=1337):
    label_paths, img_paths = get_paths(fraction=1, seed=seed, bands=["SWIRP.png"])

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