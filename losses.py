import torch.nn as nn
import torch.nn.functional as F
import torch


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, labels):
        #
        if preds.dim() == 4:
            preds = torch.sigmoid(preds)


        preds = preds.contiguous().view(-1)
        labels = labels.contiguous().view(-1)

        intersection = (preds * labels).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + labels.sum() + self.smooth)

   
        loss = 1 - dice
        return loss
    
    
class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=4): 
        super().__init__()
        self.classes = n_classes

    def to_one_hot(self, tensor):
        tensor = tensor.long() 
        n, c, h, w = tensor.size()  
        one_hot = torch.zeros(n, self.classes, h, w).to(tensor.device)
        one_hot.scatter_(1, tensor, 1)
        return one_hot

    def forward(self, inputs, target):
       
        N = inputs.size()[0]
        inputs = F.softmax(inputs, dim=1)
        
      
        target_oneHot = self.to_one_hot(target)
        inter = inputs * target_oneHot
        inter = inter.view(N, self.classes, -1).sum(2)

  
        union = inputs + target_oneHot - (inputs * target_oneHot)
        union = union.view(N, self.classes, -1).sum(2)

        loss = inter / union

        return 1 - loss.mean()

