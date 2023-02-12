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

class UNet(nn.Module):
    """
    In encoding, network makes 5 features(enc1, enc2, enc3, enc4, bottom).
    Bottom is merged with encs(enc4, enc3, enc2, enc1 in order) and enters decoding process.
    It is restored to the image size with positional features through a series of upconv layers.
    """
    def __init__(self, num_classes):
        """
        num_classes means the number of classes to classify.
        """
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
        self.dec1 = nn.Sequential(
                        Conv_block(in_channels=1024, out_channels=512),
                        Conv_block(in_channels=512, out_channels=256))

        self.upconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
                        Conv_block(in_channels=512, out_channels=256),
                        Conv_block(in_channels=256, out_channels=128))

        self.upconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
                        Conv_block(in_channels=256, out_channels=128),
                        Conv_block(in_channels=128, out_channels=64))

        self.upconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
                        Conv_block(in_channels=128, out_channels=64),
                        Conv_block(in_channels=64, out_channels=64))
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # (B, C, H, W) -> (B, C, H/2, W/2). Contracting image size to extract features.
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
        
        out = self.pool(encode_4) # (B, 512, H/16, W/16)
        out = self.bottom(out) # (B, 512, H/16, W/16)
        
        out = self.upconv1(out) # (B, 512, H/8, W/8)
        out = torch.cat((encode_4, out), dim=1) # (B, 1024, H/8, W/8)
        out = self.dec1(out) # (B, 256, H/8, W/8)
        
        out = self.upconv2(out) # (B, 256, H/4, W/4)
        out = torch.cat((encode_3, out), dim=1) # (B, 512, H/4, W/4)
        out = self.dec2(out) # (B, 128, H/4, W/4)
        
        out = self.upconv3(out) # (B, 128, H/2, W/2)
        out = torch.cat((encode_2, out), dim=1) # (B, 256, H/2, W/2)
        out = self.dec3(out) # (B, 64, H/2, W/2)
        
        out = self.upconv4(out) # (B, 64, H, W)
        out = torch.cat((encode_1, out), dim=1) # (B, 128, H, W)
        out = self.dec4(out) # (B, 64, H, W)
        out = self.fc(out) # (B, 1, H, W)
        out = torch.sigmoid(out)
        
        return out