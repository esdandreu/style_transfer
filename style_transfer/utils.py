import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10,10)
mpl.rcParams['axes.grid'] = False

import numpy as np
import re

from PIL import Image
from pathlib import Path
from typing import Union, List
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
    
_block_pattern = re.compile(
    r'block(?P<block>[0-9]{1})_'
    r'(?P<layer>conv(?P<conv>[0-9]{1})|(?P<pool>[p]{1})ool)'
    )

def append_codename(layers: List[str]):
    """VGG19 is composed of the following layers
    [
        'input_2', 'block1_conv1', 'block1_conv2', 'block1_pool', 
        'block2_conv1', 'block2_conv2', 'block2_pool', 'block3_conv1', 
        'block3_conv2', 'block3_conv3', 'block3_conv4', 'block3_pool', 
        'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_conv4', 
        'block4_pool', 'block5_conv1', 'block5_conv2', 'block5_conv3', 
        'block5_conv4', 'block5_pool'
    ]
    """
    b = []
    l =[]
    for layer in layers:
        match = _block_pattern.match(layer)
        if match:
            match = match.groupdict()
            b.append(match['block'])
            l.append(match['conv'] if 'conv' in match else match['pool'])
        else:
            raise ValueError(f'"{layer}" is not an accepted layer value')
    return (f'{len(layers)}_B{"".join(b)}_L{"".join(l)}', layers)