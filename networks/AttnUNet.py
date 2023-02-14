import torch
import torch.nn as nn


class Conv_block(nn.Module):
    """
    Create conv block with convoluion layer, batchnormalization and relu function to use in U-Net network.
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3, stride:int=1, padding:int=1, padding_mode="reflect", bias=True): # To make the output size equal input size, set padding to 1.
        """
        output size = (input size - kernel size + 2 * padding) / stride + 1
        In this case, since kernel size = 3, padding = 1 and stride = 1, output size = input size
        """        
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode) # (B, in_channels, H, W) -> (B, out_channels, H, W)
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x

class Attention(nn.Module):
    def __init__(self, f_g, f_x, f_int):
        """
        x is skipped vector from encoding and g comes from previous layer.
        g and x are concatenated after conv operation each other.
        After that, g+x goes through relu function and conv operation again.
        Finally, features are extracted via sigmoid.
        """
        super().__init__()
        self.w_g = nn.Sequential(
                        nn.Conv2d(in_channels=f_g, out_channels=f_int, kernel_size=1, stride=1, padding=0, bias=True),
                        nn.BatchNorm2d(num_features=f_int))
        self.w_x = nn.Sequential(
                        nn.Conv2d(in_channels=f_x, out_channels=f_int, kernel_size=1, stride=1, padding=0, bias=True),
                        nn.BatchNorm2d(num_features=f_int))
        self.psi = nn.Sequential(
                        nn.Conv2d(in_channels=f_int, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
                        nn.BatchNorm2d(num_features=1),
                        nn.Sigmoid())
        self.relu = nn.ReLU()
    
    def forward(self, g, x):
        g = self.w_g(g)
        x = self.w_x(x)
        psi = self.relu(g+x)
        psi = self.psi(psi)
        
        return psi*x
    
class Attention_UNet(nn.Module):
    """
    Likewise U-Net, network makes 5 features(enc1, enc2, enc3, enc4, bottom).
    But in AttnU-Net, bottom is merged with encs passed through attention mechanism and enters decoding process.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes            
        
        self.enc1 = nn.Sequential(
                        Conv_block(in_channels=3, out_channels=64),
                        Conv_block(in_channels=64, out_channels=64))
        self.enc2 = nn.Sequential(
                        Conv_block(in_channels=64, out_channels=128),
                        Conv_block(in_channels=128, out_channels=128))
        self.enc3 = nn.Sequential(
                        Conv_block(in_channels=128, out_channels=256),
                        Conv_block(in_channels=256, out_channels=256))
        self.enc4 = nn.Sequential(
                        Conv_block(in_channels=256, out_channels=512),
                        Conv_block(in_channels=512, out_channels=512))
        
        self.bottom = nn.Sequential(
                        Conv_block(in_channels=512, out_channels=1024),
                        Conv_block(in_channels=1024, out_channels=512))
        
        self.upconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2)
        self.att1 = Attention(f_g=512, f_x=512, f_int=512)
        self.dec1 = nn.Sequential(
                        Conv_block(in_channels=1024, out_channels=512),
                        Conv_block(in_channels=512, out_channels=256))
        self.upconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        self.att2 = Attention(f_g=256, f_x=256, f_int=256)
        self.dec2 = nn.Sequential(
                        Conv_block(in_channels=512, out_channels=256),
                        Conv_block(in_channels=256, out_channels=128))
        self.upconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2)
        self.att3 = Attention(f_g=128, f_x=128, f_int=128)
        self.dec3 = nn.Sequential(
                        Conv_block(in_channels=256, out_channels=128),
                        Conv_block(in_channels=128, out_channels=64))
        self.upconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2)
        self.att4 = Attention(f_g=64, f_x=64, f_int=64)
        self.dec4 = nn.Sequential(
                        Conv_block(in_channels=128, out_channels=64),
                        Conv_block(in_channels=64, out_channels=64))        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        
    def forward(self, x):
        # input = (B, 3, H, W)
        encode_1 = self.enc1(x) # (B, 64, H, W)
        
        encode_2 = self.pool(encode_1) # (B, 64, H/2, W/2)
        encode_2 = self.enc2(encode_2) # (B, 128, H/2, W/2)
        
        encode_3 = self.pool(encode_2) # (B, 128, H/4, W/4)
        encode_3 = self.enc3(encode_3) # (B, 256, H/4, W/4)
        
        encode_4 = self.pool(encode_3) # (B, 256, H/8, W/8)
        encode_4 = self.enc4(encode_4) # (B, 512, H/8, W/8)
        
        bot = self.pool(encode_4) # (B, 512, H/16, W/16)
        bot = self.bottom(bot) # (B, 512, H/16, W/16)
        
        decode_1 = self.upconv1(bot) # (B, 512, H/8, W/8)
        attention_1 = self.att1(g=decode_1, x=encode_4) # (B, 512, H/8, W/8)
        decode_1 = torch.cat((attention_1, decode_1), dim=1) # (B, 1024, H/8, W/8)
        decode_1 = self.dec1(decode_1) # (B, 256, H/8, W/8)
        
        decode_2 = self.upconv2(decode_1) # (B, 256, H/4, W/4)
        attention_2 = self.att2(g=decode_2, x=encode_3) # (B, 256, H/4, W/4)
        decode_2 = torch.cat((attention_2, decode_2), dim=1) # (B, 512, H/4, W/4)
        decode_2 = self.dec2(decode_2) # (B, 128, H/4, W/4)
        
        decode_3 = self.upconv3(decode_2) # (B, 128, H/2, W/2)
        attention_3 = self.att3(g=decode_3, x=encode_2) # (B, 128, H/2, W/2)
        decode_3 = torch.cat((attention_3, decode_3), dim=1) # (B, 256, H/2, W/2)
        decode_3 = self.dec3(decode_3) # (B, 64, H/2, W/2)
        
        decode_4 = self.upconv4(decode_3) # (B, 64, H, W)
        attention_4 = self.att4(g=decode_4, x=encode_1) # (B, 64, H, W)
        decode_4 = torch.cat((attention_4, decode_4), dim=1) # (B, 128, H, W)
        decode_4 = self.dec4(decode_4) # (B, 64, H, W)
        out = self.fc(decode_4) # (B, 1, H, W)
        out = torch.sigmoid(out)
        return out