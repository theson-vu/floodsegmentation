from torch import nn, cat, sigmoid
import torchvision.models as models
import torch.nn.functional as F
from segmentation_models_pytorch.encoders._base import EncoderMixin
from segmentation_models_pytorch.encoders.resnet import ResNetEncoder
import segmentation_models_pytorch as smp
import torch
import pywt
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    
class DualResNetEncoder(nn.Module, smp.encoders._base.EncoderMixin):
    def __init__(self, in_channels, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._in_channels = in_channels  # Required for EncoderMixin
        self._out_channels = [self._in_channels, 128, 384, 768, 1536, 3072]  # Output channels at each stage

        # Initialize ResNet50 encoder from segmentation_models_pytorch
        # Pass in_channels=3 for RGB images
        self.resnet_encoder = smp.encoders.get_encoder(
            name='resnet50',            # ResNet-50 architecture
            in_channels=3,              # Number of input channels for ResNet (e.g., 3 for RGB)
            depth=self._depth,          # Depth of the encoder (number of stages to use)
            weights='imagenet'          # Use ImageNet pre-trained weights (optional)
        )

        self.img_enc_conv_01 = self.residual_block(self._in_channels, 64)
        self.img_enc_conv_02 = self.residual_block(64, 128)
        self.img_enc_conv_03 = self.residual_block(128, 256)
        self.img_enc_conv_04 = self.residual_block(256, 512)
        self.img_enc_conv_05 = self.residual_block(512, 1024)
        # Downsampling
        self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Pass the first 3 channels (RGB) through the ResNet encoder
        img_layers = []
        resnet_layers = self.resnet_encoder(x[:, :3, :, :])  # RGB input for ResNet
        img_enc_1 = self.img_enc_conv_01(self.down_sample(x))
        img_enc_2 = self.img_enc_conv_02(self.down_sample(img_enc_1))
        img_enc_3 = self.img_enc_conv_03(self.down_sample(img_enc_2))
        img_enc_4 = self.img_enc_conv_04(self.down_sample(img_enc_3))
        img_enc_5 = self.img_enc_conv_05(self.down_sample(img_enc_4))
        img_layers.append(img_enc_1)
        img_layers.append(img_enc_2)
        img_layers.append(img_enc_3)
        img_layers.append(img_enc_4)
        img_layers.append(img_enc_5)
        return [x] + [cat((i, j), dim=1) for i, j in zip(img_layers, resnet_layers[1:])]

    @staticmethod
    def residual_block(in_channels, out_channels):
        return ResidualBlock(in_channels, out_channels)


class DualEncoder(nn.Module, EncoderMixin):
    def __init__(self, img_channels, dft_channels, depth=4, **kwargs):
        super().__init__(**kwargs)        
        # Initialize the EncoderMixin to integrate with segmentation_models_pytorch
        self._depth = depth
        self.img_channels = img_channels
        self.sec_channels = dft_channels
        self._in_channels = img_channels + dft_channels  # Required for EncoderMixin
        self._out_channels =[self._in_channels, 128, 256, 512, 1024, 2048]  # Output channels at each stage
        
        # Image encoder branch
        self.img_enc_conv_01 = self.residual_block(img_channels, 64)
        self.img_enc_conv_02 = self.residual_block(64, 128)
        self.img_enc_conv_03 = self.residual_block(128, 256)
        self.img_enc_conv_04 = self.residual_block(256, 512)
        self.img_enc_conv_05 = self.residual_block(512, 1024)

        # Spectra encoder branch
        self.spec_enc_conv_01 = self.residual_block(dft_channels, 64)
        self.spec_enc_conv_02 = self.residual_block(64, 128)
        self.spec_enc_conv_03 = self.residual_block(128, 256)
        self.spec_enc_conv_04 = self.residual_block(256, 512)
        self.spec_enc_conv_05 = self.residual_block(512, 1024)
        
        # Downsampling
        self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x_img):
        # Process the original image encoder
        img_enc_1 = self.img_enc_conv_01(self.down_sample(x_img[:, :self.img_channels, :, :]))
        img_enc_2 = self.img_enc_conv_02(self.down_sample(img_enc_1))
        img_enc_3 = self.img_enc_conv_03(self.down_sample(img_enc_2))
        img_enc_4 = self.img_enc_conv_04(self.down_sample(img_enc_3))
        img_enc_5 = self.img_enc_conv_05(self.down_sample(img_enc_4))

        # Process the spectra encoder
        spec_enc_1 = self.spec_enc_conv_01(self.down_sample(x_img[:, -self.sec_channels:, :, :]))
        spec_enc_2 = self.spec_enc_conv_02(self.down_sample(spec_enc_1))
        spec_enc_3 = self.spec_enc_conv_03(self.down_sample(spec_enc_2))
        spec_enc_4 = self.spec_enc_conv_04(self.down_sample(spec_enc_3))
        spec_enc_5 = self.spec_enc_conv_05(self.down_sample(spec_enc_4))

        # Combine feature maps from both encoders at each stage
        combined_enc_1 = cat((img_enc_1, spec_enc_1), dim=1)
        combined_enc_2 = cat((img_enc_2, spec_enc_2), dim=1)
        combined_enc_3 = cat((img_enc_3, spec_enc_3), dim=1)
        combined_enc_4 = cat((img_enc_4, spec_enc_4), dim=1)
        combined_enc_5 = cat((img_enc_5, spec_enc_5), dim=1)
        # Return all encoder features for use in the decoder (skip connections)
        return [x_img, combined_enc_1, combined_enc_2, combined_enc_3, combined_enc_4, combined_enc_5]

    @staticmethod
    def residual_block(in_channels, out_channels):
        return ResidualBlock(in_channels, out_channels)


class SingleEncoder(nn.Module, EncoderMixin):
    def __init__(self, img_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._in_channels = img_channels  # Required for EncoderMixin
        self._out_channels =[self._in_channels, 64, 128, 256, 512, 1024]  # Output channels at each stage
        self.img_enc_conv_01 = self.residual_block(img_channels, 64)
        self.img_enc_conv_02 = self.residual_block(64, 128)
        self.img_enc_conv_03 = self.residual_block(128, 256)
        self.img_enc_conv_04 = self.residual_block(256, 512)
        self.img_enc_conv_05 = self.residual_block(512, 1024)

        # Downsampling
        self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2)
        # Upsampling
        self.up_sample = nn.ConvTranspose2d(in_channels=24, out_channels=24, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x_img):
        if self._in_channels == 8:
            img_enc_1 = self.img_enc_conv_01(x_img)
        else:
            img_enc_1 = self.img_enc_conv_01(self.down_sample(x_img))
        img_enc_2 = self.img_enc_conv_02(self.down_sample(img_enc_1))
        img_enc_3 = self.img_enc_conv_03(self.down_sample(img_enc_2))
        img_enc_4 = self.img_enc_conv_04(self.down_sample(img_enc_3))
        img_enc_5 = self.img_enc_conv_05(self.down_sample(img_enc_4))

        return [x_img, img_enc_1, img_enc_2, img_enc_3, img_enc_4, img_enc_5]
    
    @staticmethod
    def residual_block(in_channels, out_channels):
        return ResidualBlock(in_channels, out_channels)
    

class SingleEncoder2(nn.Module, EncoderMixin):
    def __init__(self, img_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._in_channels = img_channels  # Required for EncoderMixin
        self._out_channels = [self._in_channels, 64, 128, 256, 512, 1024]  # Output channels at each stage
        
        # Convolutional blocks with residual connections
        self.img_enc_conv_01 = self.residual_block(img_channels, 64)
        self.img_enc_conv_02 = self.residual_block(64, 128)
        self.img_enc_conv_03 = self.residual_block(128, 256)
        self.img_enc_conv_04 = self.residual_block(256, 512)
        self.img_enc_conv_05 = self.residual_block(512, 1024)

        # Downsampling layer
        self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x_img):
        # Process each encoding layer: convolution first, downsampling after
        img_enc_1 = self.img_enc_conv_01(x_img)               # Apply the first convolution
        img_enc_2 = self.img_enc_conv_02(self.down_sample(img_enc_1))   # Downsample after convolution
        img_enc_3 = self.img_enc_conv_03(self.down_sample(img_enc_2))
        img_enc_4 = self.img_enc_conv_04(self.down_sample(img_enc_3))
        img_enc_5 = self.img_enc_conv_05(self.down_sample(img_enc_4))

        return [x_img, img_enc_1, img_enc_2, img_enc_3, img_enc_4, img_enc_5]
    
    @staticmethod
    def residual_block(in_channels, out_channels):
        return ResidualBlock(in_channels, out_channels)

class EncoderWav(nn.Module, EncoderMixin):
    def __init__(self, img_channels, depth=4, single=False, **kwargs):
        super().__init__(**kwargs)
        self.single = single
        self._depth = depth
        self._in_channels = img_channels  # Required for EncoderMixin
        self._out_channels =[self._in_channels * (self.single * 4), 64, 128, 256, 512, 1024]  # Output channels at each stage

        if self.single:
            self.img_enc_conv_01 = self.residual_block(img_channels * 4, 64)
        else:
            self.img_enc_conv_01 = self.residual_block(img_channels + img_channels * 4, 64)
        self.img_enc_conv_02 = self.residual_block(64 + img_channels * 4, 128)
        self.img_enc_conv_03 = self.residual_block(128 + img_channels * 4, 256)
        self.img_enc_conv_04 = self.residual_block(256 + img_channels * 4, 512)
        self.img_enc_conv_05 = self.residual_block(512 + img_channels * 4, 1024)

        # Downsampling
        self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2)

    def apply_full_wavelet_transform(self, input_tensor):
        batch_size, num_channels, height, width = input_tensor.shape
        transformed_channels = []
        LL_ = []

        for i in range(batch_size):
            for j in range(num_channels):
                # Extract single channel and apply wavelet transform
                single_channel = input_tensor[i, j, :, :].cpu().detach().numpy()
                coeffs2 = pywt.dwt2(single_channel, "haar")
                LL, (LH, HL, HH) = coeffs2

                # Convert components to tensors on the input device
                LL_tensor = torch.tensor(LL, dtype=torch.float32, device=input_tensor.device)
                LH_tensor = torch.tensor(LH, dtype=torch.float32, device=input_tensor.device)
                HL_tensor = torch.tensor(HL, dtype=torch.float32, device=input_tensor.device)
                HH_tensor = torch.tensor(HH, dtype=torch.float32, device=input_tensor.device)

                # Concatenate components into a single tensor for this channel
                transformed = torch.stack([LL_tensor, LH_tensor, HL_tensor, HH_tensor], dim=0)

                # Append to lists
                LL_.append(LL_tensor)
                transformed_channels.append(transformed)

        # Stack and reshape the transformed components
        transformed_tensor = torch.stack(transformed_channels).view(batch_size, num_channels * 4, LL_tensor.shape[0], LL_tensor.shape[1])
        transformed_LL = torch.stack(LL_).view(batch_size, num_channels, LL_tensor.shape[0], LL_tensor.shape[1])

        return transformed_LL, transformed_tensor
    

    def forward(self, x_img):
        # Create wavelets      
        LL_1, wav_1 = self.apply_full_wavelet_transform(x_img)
        LL_2, wav_2 = self.apply_full_wavelet_transform(LL_1)
        LL_3, wav_3 = self.apply_full_wavelet_transform(LL_2)
        LL_4, wav_4 = self.apply_full_wavelet_transform(LL_3)
        _, wav_5 = self.apply_full_wavelet_transform(LL_4)

        # Encoder
        if self.single:
            x_img = wav_1
            img_enc_1 = self.img_enc_conv_01(wav_1)
            img_enc_2 = self.img_enc_conv_02(cat((self.down_sample(img_enc_1), wav_2), dim=1))
            img_enc_3 = self.img_enc_conv_03(cat((self.down_sample(img_enc_2), wav_3), dim=1))
            img_enc_4 = self.img_enc_conv_04(cat((self.down_sample(img_enc_3), wav_4), dim=1))
            img_enc_5 = self.img_enc_conv_05(cat((self.down_sample(img_enc_4), wav_5), dim=1))
        else:
            img_enc_1 = self.img_enc_conv_01(cat((self.down_sample(x_img), wav_1), dim=1))
            img_enc_2 = self.img_enc_conv_02(cat((self.down_sample(img_enc_1), wav_2), dim=1))
            img_enc_3 = self.img_enc_conv_03(cat((self.down_sample(img_enc_2), wav_3), dim=1))
            img_enc_4 = self.img_enc_conv_04(cat((self.down_sample(img_enc_3), wav_4), dim=1))
            img_enc_5 = self.img_enc_conv_05(cat((self.down_sample(img_enc_4), wav_5), dim=1))
        
        #print(x_img.shape, img_enc_1.shape, img_enc_2.shape, img_enc_3.shape, img_enc_4.shape, img_enc_5.shape)

        return [x_img, img_enc_1, img_enc_2, img_enc_3, img_enc_4, img_enc_5]
    
    @staticmethod
    def residual_block(in_channels, out_channels):
        return ResidualBlock(in_channels, out_channels)


class DualEncoderWav(nn.Module, EncoderMixin):
    def __init__(self, img_channels, dft_channels, depth=4, **kwargs):
        super().__init__(**kwargs)        
        # Initialize the EncoderMixin to integrate with segmentation_models_pytorch
        self._depth = depth
        self.img_channels = img_channels
        self.sec_channels = dft_channels
        self.wav = None
        self._in_channels = img_channels + dft_channels  # Required for EncoderMixin
        self._out_channels =[self._in_channels, 128, 256, 512, 1024, 2048]  # Output channels at each stage
        
        # Image encoder branch
        self.img_enc_conv_01 = self.residual_block(img_channels, 64)
        self.img_enc_conv_02 = self.residual_block(64, 128)
        self.img_enc_conv_03 = self.residual_block(128, 256)
        self.img_enc_conv_04 = self.residual_block(256, 512)
        self.img_enc_conv_05 = self.residual_block(512, 1024)

        # Spectra encoder branch
        self.spec_enc_conv_01 = self.residual_block(dft_channels, 64)
        self.spec_enc_conv_02 = self.residual_block(64, 128)
        self.spec_enc_conv_03 = self.residual_block(128, 256)
        self.spec_enc_conv_04 = self.residual_block(256, 512)
        self.spec_enc_conv_05 = self.residual_block(512, 1024)
        
        # Downsampling
        self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2)

        # Upsampling
        #self.up_sample = nn.ConvTranspose2d(in_channels=24, out_channels=24, kernel_size=4, stride=2, padding=1)

    def forward(self, x_img):
        # Process the original image encoder
        img_enc_1 = self.img_enc_conv_01(self.down_sample(x_img))
        img_enc_2 = self.img_enc_conv_02(self.down_sample(img_enc_1))
        img_enc_3 = self.img_enc_conv_03(self.down_sample(img_enc_2))
        img_enc_4 = self.img_enc_conv_04(self.down_sample(img_enc_3))
        img_enc_5 = self.img_enc_conv_05(self.down_sample(img_enc_4))

        # Process the spectra encoder
        spec_enc_1 = self.spec_enc_conv_01(self.wav)
        spec_enc_2 = self.spec_enc_conv_02(self.down_sample(spec_enc_1))
        spec_enc_3 = self.spec_enc_conv_03(self.down_sample(spec_enc_2))
        spec_enc_4 = self.spec_enc_conv_04(self.down_sample(spec_enc_3))
        spec_enc_5 = self.spec_enc_conv_05(self.down_sample(spec_enc_4))

        # Combine feature maps from both encoders at each stage
        combined_enc_1 = cat((img_enc_1, spec_enc_1), dim=1)
        combined_enc_2 = cat((img_enc_2, spec_enc_2), dim=1)
        combined_enc_3 = cat((img_enc_3, spec_enc_3), dim=1)
        combined_enc_4 = cat((img_enc_4, spec_enc_4), dim=1)
        combined_enc_5 = cat((img_enc_5, spec_enc_5), dim=1)
        # Return all encoder features for use in the decoder (skip connections)
        return [x_img, combined_enc_1, combined_enc_2, combined_enc_3, combined_enc_4, combined_enc_5]

    @staticmethod
    def residual_block(in_channels, out_channels):
        return ResidualBlock(in_channels, out_channels)


# Create encoder initializations. Mostly for naming and channel adaptations.
smp.encoders.encoders["dual_encoder_dft"] = {
    "encoder": DualEncoder,
    "pretrained_settings": None,
    "params": {
        "img_channels": 6,
        "dft_channels": 12,
    },
}

smp.encoders.encoders["dual_encoder_wav"] = {
    "encoder": EncoderWav,
    "pretrained_settings": None,
    "params": {
        "img_channels": 6,
    },
}

smp.encoders.encoders["single_encoder_6"] = {
    "encoder": SingleEncoder,
    "pretrained_settings": None,
    "params": {
        "img_channels": 6
    },
}

smp.encoders.encoders["single_encoder_24"] = {
    "encoder": EncoderWav,
    "pretrained_settings": None,
    "params": {
        "img_channels": 6,
        "depth":4,
        "single": True,
    },
}


smp.encoders.encoders["single_encoder_12"] = {
    "encoder": SingleEncoder,
    "pretrained_settings": None,
    "params": {
        "img_channels": 12
    },
}

smp.encoders.encoders["dual_resnet_encoder"] = {
    "encoder": DualResNetEncoder,  # Reference to the class
    "pretrained_settings": {         # Define any pretrained settings
        "imagenet": {
            "url": None,            # URL for pretrained weights (if applicable)
            "input_space": "RGB",   # Color space of input images
            "input_range": [0, 1],  # Normalization range
            "mean": [0.485, 0.456, 0.406],  # Mean for normalization
            "std": [0.229, 0.224, 0.225],    # Standard deviation for normalization
        }
    },
    "params": {
        "depth": 5,                # Depth of the encoder
        "in_channels": 6,
        "out_channels": (6, 64+3, 256+64, 512+256, 1024+512, 2048+1024),
    },
}

smp.encoders.encoders["dual_encoder_dft.2"] = {
    "encoder": DualEncoder,
    "pretrained_settings": None,
    "params": {
        "img_channels": 2,
        "dft_channels": 4,
    },
}

smp.encoders.encoders["dual_encoder_wav.2"] = {
    "encoder": EncoderWav,
    "pretrained_settings": None,
    "params": {
        "img_channels": 2,
    },
}

smp.encoders.encoders["single_encoder_6.2"] = {
    "encoder": SingleEncoder,
    "pretrained_settings": None,
    "params": {
        "img_channels": 2
    },
}

smp.encoders.encoders["single_encoder_24.2"] = {
    "encoder": EncoderWav,
    "pretrained_settings": None,
    "params": {
        "img_channels": 2,
        "depth":4,
        "single": True,
    },
}


smp.encoders.encoders["single_encoder_12.2"] = {
    "encoder": SingleEncoder,
    "pretrained_settings": None,
    "params": {
        "img_channels": 4
    },
}

smp.encoders.encoders["dual_resnet_encoder.2"] = {
    "encoder": DualResNetEncoder,  # Reference to the class
    "pretrained_settings": {         # Define any pretrained settings
        "imagenet": {
            "url": None,            # URL for pretrained weights (if applicable)
            "input_space": "RGB",   # Color space of input images
            "input_range": [0, 1],  # Normalization range
            "mean": [0.485, 0.456, 0.406],  # Mean for normalization
            "std": [0.229, 0.224, 0.225],    # Standard deviation for normalization
        }
    },
    "params": {
        "depth": 5,                # Depth of the encoder
        "in_channels": 3,
        "out_channels": (6, 64+3, 256+64, 512+256, 1024+512, 2048+1024),
    },
}