from torch import nn,cat,sigmoid
import numpy as np

class Unet(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.enc_conv_01 = Unet.conv_block(in_channels,32)
        self.down_sample_01 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  
        self.enc_conv_02 = Unet.conv_block(32,64)
        self.down_sample_02 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  
        self.enc_conv_03 = Unet.conv_block(64,128)
        self.down_sample_03 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  
        self.enc_conv_04 = Unet.conv_block(128,256)
        self.down_sample_04 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  
        self.base = Unet.conv_block(256,512)
        self.up_sample_04 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0)
        self.dec_conv_04 = Unet.conv_block(512,256)
        self.up_sample_03 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0)
        self.dec_conv_03 = Unet.conv_block(256,128)
        self.up_sample_02 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.dec_conv_02 = Unet.conv_block(128,64)
        self.up_sample_01 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=0)
        self.dec_conv_01 = Unet.conv_block(64,32)
        self.final_conv = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        enc_conv_1 = self.enc_conv_01(x)
        enc_conv_2 = self.enc_conv_02(self.down_sample_01(enc_conv_1))
        enc_conv_3 = self.enc_conv_03(self.down_sample_02(enc_conv_2))
        enc_conv_4 = self.enc_conv_04(self.down_sample_03(enc_conv_3))
        base_block = self.base(self.down_sample_04(enc_conv_4))
        dec_conv_4 = self.up_sample_04(base_block)
        dec_conv_4 = cat((enc_conv_4,dec_conv_4),dim=1)
        dec_conv_4 = self.dec_conv_04(dec_conv_4)
        dec_conv_3 = self.up_sample_03(dec_conv_4)
        dec_conv_3 = cat((enc_conv_3,dec_conv_3),dim=1)
        dec_conv_3 = self.dec_conv_03(dec_conv_3)
        dec_conv_2 = self.up_sample_02(dec_conv_3)
        dec_conv_2 = cat((enc_conv_2,dec_conv_2),dim=1)
        dec_conv_2 = self.dec_conv_02(dec_conv_2)        
        dec_conv_1 = self.up_sample_01(dec_conv_2)
        dec_conv_1 = cat((enc_conv_1,dec_conv_1),dim=1)
        dec_conv_1 = self.dec_conv_01(dec_conv_1)
        return sigmoid(self.final_conv(dec_conv_1))
        
    @staticmethod  
    def conv_block(_in,_out):
        model = nn.Sequential(
            nn.Conv2d(in_channels=_in, out_channels=_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=_out, out_channels=_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        return(model)
    

    

class UnetLarge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.enc_conv_01 = UnetLarge.conv_block(in_channels, 32)
        self.down_sample_01 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  
        self.enc_conv_02 = UnetLarge.conv_block(32, 64)
        self.down_sample_02 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  
        self.enc_conv_03 = UnetLarge.conv_block(64, 128)
        self.down_sample_03 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.enc_conv_04 = UnetLarge.conv_block(128, 256)
        self.down_sample_04 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.enc_conv_05 = UnetLarge.conv_block(256, 512)
        self.down_sample_05 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) 
        
        self.base = UnetLarge.conv_block(512, 1024)
        
        self.up_sample_05 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0)
        self.dec_conv_05 = UnetLarge.conv_block(1024, 512)
        self.up_sample_04 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0)
        self.dec_conv_04 = UnetLarge.conv_block(512, 256)
        self.up_sample_03 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0)
        self.dec_conv_03 = UnetLarge.conv_block(256, 128)
        self.up_sample_02 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.dec_conv_02 = UnetLarge.conv_block(128, 64)
        self.up_sample_01 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=0)
        self.dec_conv_01 = UnetLarge.conv_block(64, 32)
        self.final_conv = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        enc_conv_1 = self.enc_conv_01(x)
        enc_conv_2 = self.enc_conv_02(self.down_sample_01(enc_conv_1))
        enc_conv_3 = self.enc_conv_03(self.down_sample_02(enc_conv_2))
        enc_conv_4 = self.enc_conv_04(self.down_sample_03(enc_conv_3))
        enc_conv_5 = self.enc_conv_05(self.down_sample_04(enc_conv_4))
        
        base_block = self.base(self.down_sample_05(enc_conv_5))

        dec_conv_5 = self.up_sample_05(base_block)
        dec_conv_5 = cat((enc_conv_5, dec_conv_5), dim=1)
        dec_conv_5 = self.dec_conv_05(dec_conv_5)
        
        dec_conv_4 = self.up_sample_04(dec_conv_5)
        dec_conv_4 = cat((enc_conv_4, dec_conv_4), dim=1)
        dec_conv_4 = self.dec_conv_04(dec_conv_4)
        
        dec_conv_3 = self.up_sample_03(dec_conv_4)
        dec_conv_3 = cat((enc_conv_3, dec_conv_3), dim=1)
        dec_conv_3 = self.dec_conv_03(dec_conv_3)
        
        dec_conv_2 = self.up_sample_02(dec_conv_3)
        dec_conv_2 = cat((enc_conv_2, dec_conv_2), dim=1)
        dec_conv_2 = self.dec_conv_02(dec_conv_2)        
        
        dec_conv_1 = self.up_sample_01(dec_conv_2)
        dec_conv_1 = cat((enc_conv_1, dec_conv_1), dim=1)
        dec_conv_1 = self.dec_conv_01(dec_conv_1)
        
        return sigmoid(self.final_conv(dec_conv_1))
        
    @staticmethod  
    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
class Unet6(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.enc_conv_01 = Unet6.conv_block(in_channels, 32)
        self.down_sample_01 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  
        self.enc_conv_02 = Unet6.conv_block(32, 64)
        self.down_sample_02 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  
        self.enc_conv_03 = Unet6.conv_block(64, 128)
        self.down_sample_03 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.enc_conv_04 = Unet6.conv_block(128, 256)
        self.down_sample_04 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.enc_conv_05 = Unet6.conv_block(256, 512)
        self.down_sample_05 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.enc_conv_06 = Unet6.conv_block(512, 1024)
        self.down_sample_06 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.base = Unet6.conv_block(1024, 2048)
        
        self.up_sample_06 = nn.ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=2, stride=2, padding=0)
        self.dec_conv_06 = Unet6.conv_block(2048, 1024)
        self.up_sample_05 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0)
        self.dec_conv_05 = Unet6.conv_block(1024, 512)
        self.up_sample_04 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0)
        self.dec_conv_04 = Unet6.conv_block(512, 256)
        self.up_sample_03 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0)
        self.dec_conv_03 = Unet6.conv_block(256, 128)
        self.up_sample_02 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.dec_conv_02 = Unet6.conv_block(128, 64)
        self.up_sample_01 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=0)
        self.dec_conv_01 = Unet6.conv_block(64, 32)
        self.final_conv = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        enc_conv_1 = self.enc_conv_01(x)
        enc_conv_2 = self.enc_conv_02(self.down_sample_01(enc_conv_1))
        enc_conv_3 = self.enc_conv_03(self.down_sample_02(enc_conv_2))
        enc_conv_4 = self.enc_conv_04(self.down_sample_03(enc_conv_3))
        enc_conv_5 = self.enc_conv_05(self.down_sample_04(enc_conv_4))
        enc_conv_6 = self.enc_conv_06(self.down_sample_05(enc_conv_5))
        
        base_block = self.base(self.down_sample_06(enc_conv_6))

        dec_conv_6 = self.up_sample_06(base_block)
        dec_conv_6 = cat((enc_conv_6, dec_conv_6), dim=1)
        dec_conv_6 = self.dec_conv_06(dec_conv_6)
        
        dec_conv_5 = self.up_sample_05(dec_conv_6)
        dec_conv_5 = cat((enc_conv_5, dec_conv_5), dim=1)
        dec_conv_5 = self.dec_conv_05(dec_conv_5)
        
        dec_conv_4 = self.up_sample_04(dec_conv_5)
        dec_conv_4 = cat((enc_conv_4, dec_conv_4), dim=1)
        dec_conv_4 = self.dec_conv_04(dec_conv_4)
        
        dec_conv_3 = self.up_sample_03(dec_conv_4)
        dec_conv_3 = cat((enc_conv_3, dec_conv_3), dim=1)
        dec_conv_3 = self.dec_conv_03(dec_conv_3)
        
        dec_conv_2 = self.up_sample_02(dec_conv_3)
        dec_conv_2 = cat((enc_conv_2, dec_conv_2), dim=1)
        dec_conv_2 = self.dec_conv_02(dec_conv_2)        
        
        dec_conv_1 = self.up_sample_01(dec_conv_2)
        dec_conv_1 = cat((enc_conv_1, dec_conv_1), dim=1)
        dec_conv_1 = self.dec_conv_01(dec_conv_1)
        
        return sigmoid(self.final_conv(dec_conv_1))
        
    @staticmethod  
    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

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

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ResAttUnet(nn.Module):
    def __init__(self, img_channels, spec_channels, out_channels):
        super(ResAttUnet, self).__init__()

        # Encoder for the original image
        self.img_enc_conv_01 = ResAttUnet.residual_block(img_channels, 32)
        self.img_down_sample_01 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.img_enc_conv_02 = ResAttUnet.residual_block(32, 64)
        self.img_down_sample_02 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.img_enc_conv_03 = ResAttUnet.residual_block(64, 128)
        self.img_down_sample_03 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.img_enc_conv_04 = ResAttUnet.residual_block(128, 256)
        self.img_down_sample_04 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.img_enc_conv_05 = ResAttUnet.residual_block(256, 512)
        self.img_down_sample_05 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Encoder for the amplitude and phase spectra
        self.spec_enc_conv_01 = ResAttUnet.residual_block(spec_channels, 32)
        self.spec_down_sample_01 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.spec_enc_conv_02 = ResAttUnet.residual_block(32, 64)
        self.spec_down_sample_02 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.spec_enc_conv_03 = ResAttUnet.residual_block(64, 128)
        self.spec_down_sample_03 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.spec_enc_conv_04 = ResAttUnet.residual_block(128, 256)
        self.spec_down_sample_04 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.spec_enc_conv_05 = ResAttUnet.residual_block(256, 512)
        self.spec_down_sample_05 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Base block
        self.base = ResAttUnet.residual_block(1024, 1024)  # Corrected here

        # Decoder
        self.up_sample_05 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0)
        self.dec_conv_05 = ResAttUnet.residual_block(1536, 512)  # 512 from img + 512 from spec + 512 from upsample
        self.att_05 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        
        self.up_sample_04 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0)
        self.dec_conv_04 = ResAttUnet.residual_block(768, 256)  # 256 from img + 256 from spec + 256 from upsample
        self.att_04 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        
        self.up_sample_03 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0)
        self.dec_conv_03 = ResAttUnet.residual_block(384, 128)  # 128 from img + 128 from spec + 128 from upsample
        self.att_03 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        
        self.up_sample_02 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.dec_conv_02 = ResAttUnet.residual_block(192, 64)  # 64 from img + 64 from spec + 64 from upsample
        self.att_02 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        
        self.up_sample_01 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=0)
        self.dec_conv_01 = ResAttUnet.residual_block(96, 32)  # 32 from img + 32 from spec + 32 from upsample

        self.final_conv = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x_img, x_spec):
        # Process the original image
        img_enc_conv_1 = self.img_enc_conv_01(x_img)
        img_enc_conv_2 = self.img_enc_conv_02(self.img_down_sample_01(img_enc_conv_1))
        img_enc_conv_3 = self.img_enc_conv_03(self.img_down_sample_02(img_enc_conv_2))
        img_enc_conv_4 = self.img_enc_conv_04(self.img_down_sample_03(img_enc_conv_3))
        img_enc_conv_5 = self.img_enc_conv_05(self.img_down_sample_04(img_enc_conv_4))

        # Process the amplitude and phase spectra
        spec_enc_conv_1 = self.spec_enc_conv_01(x_spec)
        spec_enc_conv_2 = self.spec_enc_conv_02(self.spec_down_sample_01(spec_enc_conv_1))
        spec_enc_conv_3 = self.spec_enc_conv_03(self.spec_down_sample_02(spec_enc_conv_2))
        spec_enc_conv_4 = self.spec_enc_conv_04(self.spec_down_sample_03(spec_enc_conv_3))
        spec_enc_conv_5 = self.spec_enc_conv_05(self.spec_down_sample_04(spec_enc_conv_4))
        # Combine feature maps from both encoders
        combined_enc_conv_5 = cat((img_enc_conv_5, spec_enc_conv_5), dim=1)
        base_block = self.base(self.img_down_sample_05(combined_enc_conv_5))

        # Decoder with attention blocks
        dec_conv_5 = self.up_sample_05(base_block)
        dec_conv_5 = cat((self.att_05(dec_conv_5, img_enc_conv_5), self.att_05(dec_conv_5, spec_enc_conv_5), dec_conv_5), dim=1)
        
        dec_conv_5 = self.dec_conv_05(dec_conv_5)

        dec_conv_4 = self.up_sample_04(dec_conv_5)
        dec_conv_4 = cat((self.att_04(dec_conv_4, img_enc_conv_4), self.att_04(dec_conv_4, spec_enc_conv_4), dec_conv_4), dim=1)
        dec_conv_4 = self.dec_conv_04(dec_conv_4)

        dec_conv_3 = self.up_sample_03(dec_conv_4)
        dec_conv_3 = cat((self.att_03(dec_conv_3, img_enc_conv_3), self.att_03(dec_conv_3, spec_enc_conv_3), dec_conv_3), dim=1)
        dec_conv_3 = self.dec_conv_03(dec_conv_3)

        dec_conv_2 = self.up_sample_02(dec_conv_3)
        dec_conv_2 = cat((self.att_02(dec_conv_2, img_enc_conv_2), self.att_02(dec_conv_2, spec_enc_conv_2), dec_conv_2), dim=1)
        dec_conv_2 = self.dec_conv_02(dec_conv_2)

        dec_conv_1 = self.up_sample_01(dec_conv_2)
        dec_conv_1 = cat((img_enc_conv_1, spec_enc_conv_1, dec_conv_1), dim=1)
        dec_conv_1 = self.dec_conv_01(dec_conv_1)

        return sigmoid(self.final_conv(dec_conv_1)), spec_enc_conv_5, img_enc_conv_5, dec_conv_2

    @staticmethod
    def residual_block(in_channels, out_channels):
        return ResidualBlock(in_channels, out_channels)
