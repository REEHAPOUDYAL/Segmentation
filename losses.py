# from .train3 import deeplabv3_encoder_decoder
# # from .train3 import pl
# # from .train3 import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# class mIoULoss(nn.Module):
#     def __init__(self, weight=None, size_average=True, n_classes=4):
#         super().__init__()
#         self.classes = n_classes

#     def to_one_hot(self, tensor):
#         n, h, w = tensor.size()
#         one_hot = torch.zeros(n, self.classes, h, w).to(tensor.device)
#         one_hot.scatter_(1, tensor.unsqueeze(1), 1)
#         return one_hot

#     def forward(self, inputs, target):
#         N = inputs.size(0)
#         inputs = F.softmax(inputs, dim=1)
#         target_oneHot = self.to_one_hot(target)
#         inter = inputs * target_oneHot
#         inter = inter.view(N, self.classes, -1).sum(2)
#         union = inputs + target_oneHot - inter
#         union = union.view(N, self.classes, -1).sum(2)
#         loss = inter / union
#         return 1 - loss.mean()

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

        # Flatten the tensors
        preds = preds.contiguous().view(-1)
        labels = labels.contiguous().view(-1)

        # Compute intersection and union
        intersection = (preds * labels).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + labels.sum() + self.smooth)

        # Dice loss is 1 - Dice coefficient
        loss = 1 - dice
        return loss
    
    
class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=4):  # Set n_classes to 4
        super().__init__()
        self.classes = n_classes

    def to_one_hot(self, tensor):
        tensor = tensor.long()  # Ensure tensor is a LongTensor
        n, c, h, w = tensor.size()  # Adjust size extraction
        one_hot = torch.zeros(n, self.classes, h, w).to(tensor.device)
        one_hot.scatter_(1, tensor, 1)
        return one_hot

    def forward(self, inputs, target):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)
        
        # Numerator Product
        target_oneHot = self.to_one_hot(target)
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator 
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        loss = inter / union

        ## Return average loss over classes and batch
        return 1 - loss.mean()

