from torch import nn,cat,sigmoid
    
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
    


class UnetSmall(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.enc_conv_01 = UnetSmall.conv_block(in_channels, 32)
        self.down_sample_01 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  
        self.enc_conv_02 = UnetSmall.conv_block(32, 64)
        self.down_sample_02 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  
        self.enc_conv_03 = UnetSmall.conv_block(64, 128)
        self.down_sample_03 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  
        
        self.base = UnetSmall.conv_block(128, 256)
        
        self.up_sample_03 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0)
        self.dec_conv_03 = UnetSmall.conv_block(256, 128)
        self.up_sample_02 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.dec_conv_02 = UnetSmall.conv_block(128, 64)
        self.up_sample_01 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=0)
        self.dec_conv_01 = UnetSmall.conv_block(64, 32)
        self.final_conv = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        enc_conv_1 = self.enc_conv_01(x)
        enc_conv_2 = self.enc_conv_02(self.down_sample_01(enc_conv_1))
        enc_conv_3 = self.enc_conv_03(self.down_sample_02(enc_conv_2))
        
        base_block = self.base(self.down_sample_03(enc_conv_3))
        
        dec_conv_3 = self.up_sample_03(base_block)
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
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
    

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
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )