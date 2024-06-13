# # # import os
# # # import pytorch_lightning as L
# # # from dataloader import AerialImageDataset
# # # from train5 import deeplabv3_encoder_decoder
# # # from torch.utils.data import DataLoader
# # # from torchvision.transforms import transforms
# # # import torch

# # # train_path = r"C:\Users\User\Downloads\Nishant\train"
# # # val_path = r"C:\Users\User\Downloads\Nishant\val"

# # # data_transform = transforms.Compose([
# # #     transforms.Resize((512, 512)),
# # #     transforms.ToTensor()
# # # ])

# # # train_dataset = AerialImageDataset(os.path.join(train_path, 'images'), os.path.join(train_path, 'masks'), transform=data_transform)
# # # val_dataset = AerialImageDataset(os.path.join(val_path, 'images'), os.path.join(val_path, 'masks'), transform=data_transform)

# # # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# # # val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# # # model = deeplabv3_encoder_decoder()

# # # # Adjust the refresh rate of the progress bar
# # # trainer = L.Trainer(max_epochs=100, progress_bar_refresh_rate=20)  # Adjust the refresh rate as needed
# # # trainer.fit(model, train_loader, val_loader)

# # # torch.save(model.state_dict(), r"C:\Users\User\Downloads\Nishant\main.py\model.pth")

# # import os
# # import pytorch_lightning as pl
# # from dataloader import AerialImageDataset
# # from train5 import deeplabv3_encoder_decoder
# # from torch.utils.data import DataLoader
# # from torchvision.transforms import transforms
# # import torch

# # train_path = r"C:\Users\User\Downloads\Nishant\train"
# # val_path = r"C:\Users\User\Downloads\Nishant\val"

# # data_transform = transforms.Compose([
# #     transforms.Resize((512, 512)),
# #     transforms.ToTensor()
# # ])

# # train_dataset = AerialImageDataset(os.path.join(train_path, 'images'), os.path.join(train_path, 'masks'), transform=data_transform)
# # val_dataset = AerialImageDataset(os.path.join(val_path, 'images'), os.path.join(val_path, 'masks'), transform=data_transform)

# # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# # val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# # model = deeplabv3_encoder_decoder()

# # # Adjust other trainer parameters as needed
# # trainer = pl.Trainer(max_epochs=100)
# # trainer.fit(model, train_loader, val_loader)

# # torch.save(model.state_dict(), r"C:\Users\User\Downloads\Nishant\main.py\model.pth")



# #running code 
# # import os
# # import pytorch_lightning as pl
# # from dataloader import AerialImageDataset
# # from train5 import deeplabv3_encoder_decoder
# # from torch.utils.data import DataLoader
# # from torchvision.transforms import transforms
# # import torch

# # train_path = r"C:\Users\User\Downloads\Nishant\train"
# # val_path = r"C:\Users\User\Downloads\Nishant\val"

# # data_transform = transforms.Compose([
# #     transforms.Resize((512, 512)),
# #     transforms.ToTensor()
# # ])

# # train_dataset = AerialImageDataset(os.path.join(train_path, 'images'), os.path.join(train_path, 'masks'), transform=data_transform)
# # val_dataset = AerialImageDataset(os.path.join(val_path, 'images'), os.path.join(val_path, 'masks'), transform=data_transform)

# # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# # val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# # model = deeplabv3_encoder_decoder()

# # # Adjust other trainer parameters as needed
# # trainer = pl.Trainer(num_sanity_val_steps=0, max_epochs=100)
# # trainer.fit(model, train_loader, val_loader)

# # torch.save(model.state_dict(), r"C:\Users\User\Downloads\Nishant\main.py\model.pth")

# import os
# import pytorch_lightning as pl
# from dataloader import AerialImageDataset
# from train5 import deeplabv3_encoder_decoder
# from torch.utils.data import DataLoader
# from torchvision.transforms import transforms
# import torch
# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# train_path = r"C:\Users\User\Downloads\Nishant\train"
# val_path = r"C:\Users\User\Downloads\Nishant\val"

# data_transform = transforms.Compose([
#     transforms.Resize((512, 512)),
#     transforms.ToTensor()
# ])

# train_dataset = AerialImageDataset(os.path.join(train_path, 'images'), os.path.join(train_path, 'masks'), transform=data_transform)
# val_dataset = AerialImageDataset(os.path.join(val_path, 'images'), os.path.join(val_path, 'masks'), transform=data_transform)

# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# model = deeplabv3_encoder_decoder()


# checkpoint_callback = ModelCheckpoint(
#     monitor='val_loss',  
#     dirpath='checkpoints',  
#     filename='best_model',  
#     save_top_k=1,  
#     mode='min' 
# )

# early_stop_callback = EarlyStopping(
#     monitor='val_loss',  
#     patience=20,  
#     verbose=True,
#     mode='min'  
# )


# trainer = pl.Trainer(
#     num_sanity_val_steps=0,
#     max_epochs=100,
#     callbacks=[checkpoint_callback, early_stop_callback]  # Pass both callbacks
# )
# trainer.fit(model, train_loader, val_loader)
# torch.save(model.state_dict(), r"C:\Users\User\Downloads\Nishant\main.py\model.pth")
import os
import pytorch_lightning as pl
from dataloader import AerialImageDataset
from train5 import deeplabv3_encoder_decoder
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

train_path = r"C:\Users\User\Downloads\Nishant\train"
val_path = r"C:\Users\User\Downloads\Nishant\val"

data_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

train_dataset = AerialImageDataset(os.path.join(train_path, 'images'), os.path.join(train_path, 'masks'), transform=data_transform)
val_dataset = AerialImageDataset(os.path.join(val_path, 'images'), os.path.join(val_path, 'masks'), transform=data_transform)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

model = deeplabv3_encoder_decoder()

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  
    dirpath='checkpoints1',  
    filename='best_model',  
    save_top_k=1,  
    mode='min'  # Save the model based on minimizing validation loss
)

early_stop_callback = EarlyStopping(
    monitor='val_loss',  
    patience=20,  
    verbose=True,
    mode='min'  
)

trainer = pl.Trainer(
    num_sanity_val_steps=0,
    max_epochs=100,
    callbacks=[checkpoint_callback, early_stop_callback]  # Pass both callbacks
)
trainer.fit(model, train_loader, val_loader)
torch.save(model.state_dict(), r"C:\Users\User\Downloads\Nishant\model.pth")
