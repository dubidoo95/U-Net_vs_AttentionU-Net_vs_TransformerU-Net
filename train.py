from networks.UNet import UNet
from networks.AttnUNet import Attention_UNet
from networks.TransUNet import Transformer_UNet
from sklearn.model_selection import train_test_split
from dataset import CustomDataset, transforms
from utils import dice_coef_metric, DiceLoss
from torch.utils.data  import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    # Assign variables
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = "./data"
    batch_size = 4 # according to the situation
    num_epochs = 3 # according to the situation
    model = UNet(num_classes=1).to(device) # UNet, Attention_UNet, Transformer_UNet
    model_name = "UNet" # according to the situation
    criterion = DiceLoss().to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Load dataset and split into train, validation, test set.
    dataset = CustomDataset(data_path, transform=transforms)
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.15, random_state=42, shuffle=True)
    train_dataset, test_dataset = train_test_split(train_dataset, test_size=0.15, random_state=42, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # To track history, save the values of loss and accuracies.
    train_loss_history = []
    train_iou_history = []
    val_loss_history = []
    val_iou_history = []
    writer = SummaryWriter()
    save_path = f'./{model_name}_best_model.pt'
    
    for epoch in range(num_epochs):
        train_losses = []
        train_ious = []
        val_losses = []
        val_ious = []
        
        # in train
        for idx, (imgs, masks) in tqdm(enumerate(train_dataloader)):
            model.train()
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            preds = model(imgs)
            train_loss = criterion(preds, masks)
            train_losses.append(train_loss.item())

            train_dice = dice_coef_metric(preds.data.cpu().numpy(), masks.data.cpu().numpy())
            train_ious.append(train_dice)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
        # in validation
        for idx, (imgs, masks) in tqdm(enumerate(val_dataloader)):
            model.eval()
            with torch.no_grad():
                imgs = imgs.to(device)
                masks = masks.to(device)
                
                preds = model(imgs)
                val_loss = criterion(preds, masks)
                val_losses.append(val_loss.item())
                
                val_dice = dice_coef_metric(preds.data.cpu().numpy(), masks.data.cpu().numpy())
                val_ious.append(val_dice)

                
        train_loss_mean = np.array(train_losses).mean()
        train_iou_mean = np.array(train_ious).mean()
        val_loss_mean = np.array(val_losses).mean()
        val_iou_mean = np.array(val_ious).mean()

        writer.add_scalar("Loss/train", train_loss_mean, epoch)
        writer.add_scalar("Acc/train", train_iou_mean, epoch)
        writer.add_scalar("Loss/val", val_loss_mean, epoch)
        writer.add_scalar("Acc/val", val_iou_mean, epoch)

        train_loss_history.append(train_loss_mean)
        train_iou_history.append(train_iou_mean)
        val_loss_history.append(val_loss_mean)
        val_iou_history.append(val_iou_mean)
        
        print("="*20)
        print(f"Epoch: {epoch+1} / {num_epochs}")
        print(f"Train Loss: {train_loss_mean:.5f}")
        print(f"Train Accuracy: {train_iou_mean:.5f}")
        print(f"Validation Loss: {val_loss_mean:.5f}")
        print(f"Validation Accuracy: {val_iou_mean:.5f}")
        print("="*20)
    
        # save the best model
        if min(val_loss_history) == val_loss_history[-1]:
            torch.save(model.state_dict(), save_path)
    writer.close()
    