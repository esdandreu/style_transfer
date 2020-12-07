import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10,10)
mpl.rcParams['axes.grid'] = False

import numpy as np

from PIL import Image
from typing import Union
from pathlib import Path
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.keras.applications.vgg19 import preprocess_input

def load_img(path_to_img):
    max_dim = 512
    img = Image.open(path_to_img)
    long_ = max(img.size)
    scale = max_dim/long_
    img = img.resize(
        (round(img.size[0]*scale), round(img.size[1]*scale)), 
        Image.ANTIALIAS
        )
    
    img = kp_image.img_to_array(img)
    
    # We need to broadcast the image array such that it has a batch dimension 
    img = np.expand_dims(img, axis=0)
    return img

def imshow(img, title=None):
    # Remove the batch dimension
    out = np.squeeze(img, axis=0)
    # Normalize for display 
    out = out.astype('uint8')
    plt.imshow(out)
    if title is not None:
        plt.title(title)
    plt.imshow(out)

def load_and_process_img(path_to_img):
    img = load_img(path_to_img)
    img = preprocess_input(img)
    return img

def get_feature_representations(model, content_path, style_path):
    """Helper function to compute our content and style feature
    representations.

    This function will simply load and preprocess both the content and style
    images from their path. Then it will feed them through the network to
    obtain the outputs of the intermediate layers. 

    Arguments: model: The model that we are using. content_path: The path to
        the content image. style_path: The path to the style image

    Returns: returns the style features and the content features. 
    """
    # Load our images in 
    content_image = load_and_process_img(content_path)
    style_image = load_and_process_img(style_path)
    
    # batch compute content and style features
    style_outputs = model(style_image)
    content_outputs = model(content_image)
    
    # Get the style and content feature representations from our model  
    style_features = [
        style_layer[0] 
        for style_layer in style_outputs[:model.num_style_layers]
        ]
    content_features = [
        content_layer[0] 
        for content_layer in content_outputs[model.num_content_layers:]
        ]
    return style_features, content_features

def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, (
        "Input to deprocess image must be an image of "
        "dimension [1, height, width, channel] or [height, width, channel]"
        )
    
    # perform the inverse of the preprocessing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x

def save_img(
    img_array, run_id: str, iteration: int, folder: Union[str,Path], 
    error: bool = False
    ):
    Image.fromarray(img_array).save(
        Path(folder,f'{"ERROR_" if error else ""}{run_id}_{iteration}.png')
        )