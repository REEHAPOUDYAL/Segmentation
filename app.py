import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from train5 import deeplabv3_encoder_decoder
import numpy as np


def load_model(model_path):
    model = deeplabv3_encoder_decoder()

    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


model_path = '/teamspace/studios/this_studio/Segmentation/model.pth'
  


model = load_model(model_path)

if model:
    
    st.title('Disease Segmentation')
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

    
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        
        data_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()]
        )
        image = data_transform(image)
        image = image.unsqueeze(0)  

        
        with torch.no_grad():
            output = model(image)

    
        color_map = {
            0: np.array([255, 34, 133]), 
            1: np.array([0, 252, 199]),   
            2: np.array([86, 0, 254]),   
            3: np.array([0, 0, 0])        
        }

        class_labels = {
            0: 'Unlabeled',
            1: 'Early Blight',
            2: 'Late Blight',
            3: 'Leaf Minor'
        }


        for k, v in class_labels.items():
            st.sidebar.markdown(f'<div style="color:rgb{tuple(color_map[k])};">{v}</div>', unsafe_allow_html=True)


        output = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()


        output_rgb = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
        for k, v in color_map.items():
            output_rgb[output == k] = v


        st.image(output_rgb, caption='Segmented Image.', use_column_width=True)
