import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join


class Embeddings(nn.Module):
    """
    Take 4-D tensor(B, C, H, W) and return 3-D tensor(B, hidden size, num patches)
    Flatten and vectorize images for input to transformer layers.
    """
    def __init__(self, img_size:int=512, patch_size:int=16, in_channels:int=3, hidden_size:int=768, dropout_rate:int=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        self.n_patches = (self.img_size // self.patch_size) * (self.img_size // self.patch_size) # 32 * 32 = 1024
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels, 
                                       out_channels=self.hidden_size,
                                       kernel_size=self.patch_size,
                                       stride=self.patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, self.hidden_size)) # (1, n_patches, hidden_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def forward(self, x):
        x = self.patch_embeddings(x) # (B, hidden_size, n_patches**(1/2), n_patches**(1/2)) = (B, 768, 32, 32)
        x = x.flatten(2) # (B, hidden_size, n_patches) = (B, 768, 1024)
        x = x.transpose(-1, -2) # (B, n_patches, hidden_size) = (B, 1024, 768)
        embeddings = x + self.position_embeddings # (B, n_patches, hidden) = (B, 1024, 768)
        embeddings = self.dropout(embeddings)
        return embeddings

class MLP(nn.Module):
    def __init__(self, hidden_size:int=768, dim:int=3072, dropout_rate:int=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.dim = dim
        self.dropout_rate = dropout_rate
        
        self.fc1 = nn.Linear(self.hidden_size, self.dim) # (B, n_patches, hidden_size) -> (B, n_patches, dim)
        self.fc2 = nn.Linear(self.dim, self.hidden_size) # (B, n_patches, dim) -> (B, n_patches, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self._init_weights()
        
    def _init_weights(self):
        """
        For better performance, when call the model initialize weights and bias.
        """
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)
        
    def forward(self, x): # (B, 1024, 768)
        output = self.fc1(x) # (B, 1024, 3072)
        output = F.gelu(output) # GELU = Gaussian error linear unit. Converges faster than relu.
        output = self.dropout(output)
        output = self.fc2(output) # (B, 1024, 768)
        output = self.dropout(output)
        return output

class Attention(nn.Module):
    """
    Make query, key, value layers.
    """
    def __init__(self, num_heads:int=12, hidden_size:int=768, attention_dropout_rate:int=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.attention_dropout_rate = attention_dropout_rate

        self.attention_head_size = self.hidden_size // self.num_heads # 768 / 12 = 64
        self.all_head_size = self.num_heads * self.attention_head_size # 64 * 12 = 768
        
        self.query = nn.Linear(self.hidden_size, self.all_head_size) # (B, n_patches, hidden_size) -> (B, n_patches, all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size) # (B, n_patches, hidden_size) -> (B, n_patches, all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size) # (B, n_patches, hidden_size) -> (B, n_patches, all_head_size)
        
        self.out = nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_dropout = nn.Dropout(self.attention_dropout_rate)
        self.proj_dropout = nn.Dropout(self.attention_dropout_rate)
        self.softmax = nn.Softmax(dim=-1)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size) # (B, n_patches) + (num_heads, attention_head_size) = (B, n_patches, num_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # (B, num_heads, n_patches, attention_head_size)
    
    def forward(self, hidden_states):
        query_layer = self.query(hidden_states) # (B, 1024, 768)
        key_layer = self.key(hidden_states) # (B, 1024, 768)
        value_layer = self.value(hidden_states) # (B, 1024, 768)
        
        query_layer = self.transpose_for_scores(query_layer) # (B, 12, 1024, 64)
        key_layer = self.transpose_for_scores(key_layer) # (B, 12, 1024, 64)
        value_layer = self.transpose_for_scores(value_layer) # (B, 12, 1024, 64)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # (B, 12, 1024, 1024)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores) # (B, 12, 1024, 1024)
        weights = attention_probs
        attention_probs = self.attention_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer) # (B, 12, 1024, 64)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # (B, 1024, 12, 64)
        context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) # (B, 1024, 768)
        context_layer = context_layer.view(*context_layer_shape) # (B, 1024, 768)
        
        output = self.out(context_layer) # (B, 1024, 768)
        output = self.proj_dropout(output)
        return output, weights

def np2th(weights, conv=False):
    """
    convert HWIO to OIHW
    """
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

class Block(nn.Module):
    """
    A transformer layer consists of layernorm, attention, layernorm and MLP with skip connection.
    """
    def __init__(self, hidden_size:int=768):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn = MLP()
        self.attn = Attention()
    
    def forward(self, x):
        h = x # for skip connection, keep x as h.
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x += h # after Attention, concatenate x and h.
        
        h = x # for skip connection, keep x as h.
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x += h # after Attention, concatenate x and h.
        return x, weights
    
    def load_from(self, weights, n_block):
        root = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[join(root, "MultiHeadDotProductAttention_1/query", "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[join(root, "MultiHeadDotProductAttention_1/key", "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[join(root, "MultiHeadDotProductAttention_1/value", "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[join(root, "MultiHeadDotProductAttention_1/out", "kernel")]).view(self.hidden_size, self.hidden_size).t()
            
            query_bias = np2th(weights[join(root, "MultiHeadDotProductAttention_1/query", "bias")]).view(-1)
            key_bias = np2th(weights[join(root, "MultiHeadDotProductAttention_1/key", "kernel")]).view(-1)
            value_bias = np2th(weights[join(root, "MultiHeadDotProductAttention_1/value", "kernel")]).view(-1)
            out_bias = np2th(weights[join(root, "MultiHeadDotProductAttention_1/out", "kernel")]).view(-1)
            
            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)
            
            mlp_weight_0 = np2th(weights[join(root, "MlpBlock_3/Dense_0", "kernel")]).t()
            mlp_weight_1 = np2th(weights[join(root, "MlpBlock_3/Dense_1", "kernel")]).t()
            mlp_bias_0 = np2th(weights[join(root, "MlpBlock_3/Dense_0", "bias")]).t()
            mlp_bias_1 = np2th(weights[join(root, "MlpBlock_3/Dense_1", "bias")]).t()
            
            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)
            
            self.attention_norm.weight.copy_(np2th(weights[join(root, "LayerNorm_0", "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[join(root, "LayerNorm_0", "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[join(root, "LayerNorm_2", "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[join(root, "LayerNorm_2", "bias")]))

class Encoder(nn.Module):
    """
    create transformer blocks consist of 12 transformer layers.
    """
    def __init__(self, hidden_size:int=768):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        
        for _ in range(12):
            new_layer = Block()
            self.layer.append(copy.deepcopy(new_layer))
            
    def forward(self, hidden_states):
        for layer_block in self.layer: # 12 repetitions of transformer layer.
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded

class Transformer(nn.Module):
    """
    Receive images and return the lowest layers to be used in decoder.
    """
    def __init__(self, img_size:int=512):
        super().__init__()
        self.img_size = img_size
        self.embeddings = Embeddings()
        self.encoder = Encoder()
        
    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids) # (B, C, H, W) -> (B, n_patches, hidden_size)
        encoded = self.encoder(embedding_output) # (B, n_patches, hidden_size)
        return encoded

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

class Decoder(nn.Module):
    """
    In downsampling, network makes 4 features(enc1, enc2, enc3, enc4).
    Unlike U-Net, Bottom comes from transformer block.
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
                        Conv_block(in_channels=768, out_channels=512), # since output of the transformer block has 768 channels.
                        Conv_block(in_channels=512, out_channels=512))
        
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
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        
    def forward(self, x, hidden_states):
        # input = (B, 3, H, W)
        encode_1 = self.enc1(x) # (B, 64, H, W)
        
        encode_2 = self.pool(encode_1) # (B, 64, H/2, W/2)
        encode_2 = self.enc2(encode_2) # (B, 128, H/2, W/2)
        
        encode_3 = self.pool(encode_2) # (B, 128, H/4, W/4)
        encode_3 = self.enc3(encode_3) # (B, 256, H/4, W/4)
        
        encode_4 = self.pool(encode_3) # (B, 256, H/8, W/8)
        encode_4 = self.enc4(encode_4) # (B, 512, H/8, W/8)
        
        # reshape from (B, n_patch, hidden) to (B, hidden, h, w)
        B, n_patches, hidden = hidden_states.size()  # (B, 1024, 768) = (B, H/16, 3*16*16)
        h, w = int(np.sqrt(n_patches)), int(np.sqrt(n_patches))
        transformed = hidden_states.permute(0, 2, 1) # (B, 3*16*16, H/16)
        transformed = x.contiguous().view(B, hidden, h, w) # (B, 768, H/16, W/16)
        
        out = self.bottom(transformed) # (B, 512, H/16, W/16)
        
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

class Transformer_UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.transformer = Transformer()
        self.decoder = Decoder(num_classes)
    
    def forward(self, x):
        hidden_states = self.transformer(x)
        out = self.decoder(x, hidden_states)
        return out