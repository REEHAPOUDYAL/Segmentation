
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from losses import DiceLoss
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt

class ASSP(nn.Module):
    def __init__(self, in_channels, out_channels=256, final_out_channels=4):
        super(ASSP, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        # 1x1 convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 convolutions with different dilation rates
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)


        self.convf = nn.Conv2d(out_channels * 5, final_out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.bnf = nn.BatchNorm2d(final_out_channels)

     
        self.adapool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # 1x1 convolution
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        # 3x3 convolution with dilation 6
        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)

        # 3x3 convolution with dilation 12
        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)

        # 3x3 convolution with dilation 18
        x4 = self.conv4(x)
        x4 = self.bn4(x4)
        x4 = self.relu(x4)

        # Global average pooling, 1x1 convolution, and upsample
        x5 = self.adapool(x)
        x5 = self.conv5(x5)
        x5 = self.bn5(x5)
        x5 = self.relu(x5)
        x5 = F.interpolate(x5, size=x4.shape[-2:], mode='bilinear')

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        # Final 1x1 convolution
        x = self.convf(x)
        x = self.bnf(x)
        x = self.relu(x)

        return x
    
class ResNet_50(nn.Module):
    def __init__(self, in_channels=3):
        super(ResNet_50, self).__init__()
        self.resnet_50 = models.resnet50(pretrained=True)


        self.resnet_50.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

     
        self.resnet_50 = nn.Sequential(*list(self.resnet_50.children())[:-2])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.resnet_50(x)
        return x


class deeplabv3_encoder_decoder(pl.LightningModule):
    def __init__(self, input_channels=3, output_channels=4): 
        super(deeplabv3_encoder_decoder, self).__init__()
        self.resnet = ResNet_50(in_channels=input_channels)
        self.aspp = ASSP(in_channels=2048, final_out_channels=4)
        self.conv = nn.Conv2d(in_channels=4, out_channels=output_channels, kernel_size=1)
        self.criterion = DiceLoss() 
    def forward(self, x):
        _, _, h, w = x.shape
        x = self.resnet(x)  
        x = self.aspp(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)  # Upsample
        x = self.conv(x) 
        return x    

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)
        iou = compute_iou(logits, masks)
        self.log('train_loss', loss)
        self.log('train_iou', iou)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)
        iou = compute_iou(logits, masks)
        self.log('val_loss', loss)
        self.log('val_iou', iou)
        return loss

    def on_train_epoch_end(self):
        avg_iou = self.trainer.callback_metrics['train_iou'].mean()
        train_loss = self.trainer.logged_metrics.get('train_loss')
        self.log('avg_train_iou', avg_iou)
        print("avg train iou",avg_iou)
        print("loss",train_loss)
       

    def on_validation_epoch_end(self):
        avg_iou = self.trainer.callback_metrics['val_iou'].mean()
        val_loss = self.trainer.logged_metrics.get('val_loss')

        self.log('avg_val_iou', avg_iou)
        print("avg val iou",avg_iou)
        print("val loss", val_loss) 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer




# def calculate_iou(logits, masks):
#     # Calculate predictions from logits
#     preds = torch.argmax(logits, dim=1)
#     # Calculate intersection and union
#     intersection = torch.sum(preds * masks)
#     union = torch.sum((preds.bool() | masks.bool()).int())
#     # Avoid division by zero
#     iou = intersection / union if union != 0 else torch.tensor(0.0)
#     return iou

def compute_iou(preds,labels,threshold = 0.5 , epsilon = torch.finfo(torch.float).eps):
    preds = torch.sigmoid(preds)
    # print("preds shape",preds.shape)
    preds = (preds>threshold).float()
    # print("preds shape123",preds.shape)
    # print("masks shape123",labels.shape)
    # print("masks shape123",np.unique(labels.cpu().numpy()))
    # plt.imshow(labels[0,:,:,:].T.cpu().numpy())
    # plt.show()
    n_classes = preds.shape[1]
    iou_per_class = []
    for i in range(n_classes):
        intersection = (preds[:,i,:,:] * labels[:,i,:,:]).sum((1,2))
        union = (preds[:,i,:,:]+ labels[:,i,:,:]).sum((1,2)) - intersection
        iou = (intersection + epsilon) / (union + epsilon)
        iou_per_class.append(iou.mean())
    iou_mean = sum(iou_per_class)/ n_classes
    return iou_mean
