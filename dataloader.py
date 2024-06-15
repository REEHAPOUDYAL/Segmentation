import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class AerialImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(self.image_dir)
        self.Hex_Classes = [
            ('Unlabeled', '#FF2285'),
            ('Early Blight','#00FCC7'),
            ('Late Blight', '#5600FE'),
            ('Leaf Minor', '#000000')
        ]

    def __len__(self):
        return len(self.images)
   
    def __getitem__(self, idx):
        
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '.png'))

        image = Image.open(img_path)
        mask = Image.open(mask_path).resize((512, 512))
     
        mask = np.array(mask)
        mask = self.encode_segmap(mask)
        mask = mask.astype(np.uint8) 
        mask = Image.fromarray(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

    def encode_segmap(self, mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16) 
        for i, (name, color) in enumerate(self.Hex_Classes):
            if mask.ndim == 3:
                label_mask[(mask[:,:,0] == int(color[1:3], 16)) & (mask[:,:,1] == int(color[3:5], 16)) & (mask[:,:,2] == int(color[5:7], 16))] = i
            elif mask.ndim == 2:
                label_mask[(mask == int(color[1:3], 16))] = i
              
        msk = np.zeros((512,512,4))
        for i in [0,1,2,3]:
            if i == 0:
                msk_ind = np.where(label_mask == i, 4, 0)
                msk[:,:,i] = msk_ind
            else:
                msk_ind = np.where(label_mask == i, i, 0)
                msk[:,:,i] = msk_ind
    
        return msk
