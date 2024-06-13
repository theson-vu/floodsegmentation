import torch
import tifffile
import numpy as np
import cv2
import glob

class Sentinel2_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_paths,
        mask_paths,
        transforms=None,
        seed=1337,
        num_augmentations=3
    ):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transforms = transforms
        self.dim = 6
        self.seed = seed
        self.num_augmentations = num_augmentations
        self.mean, self.std = self.get_dataset_mean_std()
        #self.pwater_mean, self.pwater_std = self.get_pwater_mean_std()

    def get_dataset_mean_std(self):
        """
        Computes the channelwise mean and std of the dataset
        """
        means = []
        stds = []

        for path_list in self.img_paths:
            images = []
            for path in path_list:
                if ".tif" in path:
                    img = tifffile.imread(path)
                elif ".png" in path:
                    img = cv2.imread(path) / 255.0
                    images.append(img)
            means.append(np.mean(images, axis=(0, 1, 2)))
            stds.append(np.std(images, axis=(0, 1, 2)))
        
        return means, stds
    
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
    
    def normalize_pwater(self, image, epsilon=1e-6):
        return (image - self.pwater_mean) / (self.pwater_std)

    def __getitem__(self, idx):
        # Load in image
        arr_x = []
        
        for i, path_list in enumerate(self.img_paths):
            img = cv2.imread(path_list[idx])
            img = img / 255.0
            img_norm = self.normalize_data(img, i)
            arr_x.append(img_norm)
        arr_x = np.concatenate(arr_x, axis=-1)
        #pwater = tifffile.imread(self.mask_paths[idx].replace("LabelWater.tif", f"LabelWater_jrc-gsw-occurrence.tif"))
        #pwater = self.normalize_pwater(pwater / 100.0)
        #pwater = np.expand_dims(pwater, axis=-1)
        #arr_x = np.concatenate((arr_x, pwater), axis=-1)

        sample = {"image": arr_x}
        self.dim = arr_x.shape[-1]
        # Load in label mask
        sample["mask"] = tifffile.imread(self.mask_paths[idx])

        # Apply Data Augmentation
        augmented_samples = []
        for _ in range(self.num_augmentations):
            augmented_sample = sample.copy()
            if self.transforms:
                augmented_sample = self.transforms(image=sample["image"], mask=sample["mask"])
            if augmented_sample["image"].shape[-1] < 20:
                augmented_sample["image"] = augmented_sample["image"].transpose((2, 0, 1))
            augmented_sample["image"] = torch.tensor(augmented_sample["image"], dtype=torch.float32)
            augmented_sample["mask"] = torch.tensor(augmented_sample["mask"], dtype=torch.long)
            augmented_samples.append(augmented_sample)
            
        return augmented_samples

    def __len__(self):
        return len(self.mask_paths)

            
def mask_to_img(label, color_dict):
    """Recodes a (H,W) mask to a (H,W,3) RGB image according to color_dict"""
    mutually_exclusive = np.zeros(label.shape + (3,), dtype=np.uint8)
    for key in range(1, len(color_dict.keys()) + 1):
        mutually_exclusive[label == key] = color_dict[key]
    return mutually_exclusive



def get_paths(fraction=1, bands=["SWIR.png", "SWIRP.png"], seed=1337):
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


def create_splits(train_percentage=0.6, val_percentage=0.2, test_percentage=0.1):
    label_paths, img_paths = get_paths(fraction=1)

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