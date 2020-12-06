import time
import logging

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path

from style_transfer.model import get_model
from style_transfer.loss import gram_matrix, compute_grads
from style_transfer.utils import (
    get_feature_representations, load_and_process_img, deprocess_img, save_img
    )
from style_transfer.config import (
    OUTPUT_FOLDER, STYLE_FOLDER, CONTENT_FOLDER, CHECKPOINTS_PER_RUN
    )

logger = logging.getLogger(__name__)

def run_style_transfer(
    content_name: str, 
    style_name: str,
    num_iterations: int = 1000,
    output_folder: Path = OUTPUT_FOLDER,
    content_weight: float = 1e3, 
    style_weight: float = 1e-2
    ): 

    # Get path from content_name
    # TODO solve the formats problem (get name from path instead of the other way)
    content_path = Path(CONTENT_FOLDER, f'{content_name}.jpg')
    style_path = Path(STYLE_FOLDER, f'{style_name}.jpg')

    # Image and stats identifier form input parameters
    run_id = f'{content_name}_{style_name}'

    # We don't need to (or want to) train any layers of our model, so we set
    # their trainable to false. 
    model = get_model() 
    for layer in model.layers:
        layer.trainable = False
    
    # Get the style and content feature representations (from our specified
    # intermediate layers) 
    style_features, content_features = get_feature_representations(
        model, content_path, style_path
        )
    gram_style_features = [
        gram_matrix(style_feature) for style_feature in style_features
        ]
    
    # Set initial image
    init_image = load_and_process_img(content_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)
    # Create our optimizer
    # TODO play with optimizer? 
    opt = tf.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)

    # Store our best result
    best_loss, best_img = float('inf'), None
    
    # Argument for computing the gradients
    loss_weights = (style_weight, content_weight)
        
    # ? What is this
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means   

    # Interval for checkpoints
    run_start = time.time()
    checkpoint_interval = num_iterations/(CHECKPOINTS_PER_RUN)

    # TODO create stats file
    
    try: 
        for i in range(num_iterations):
            start = time.time()
            grads, loss, style_score, content_score = compute_grads(
                model=model,
                loss_weights=loss_weights,
                init_image=init_image,
                gram_style_features=gram_style_features,
                content_features=content_features
                )
            opt.apply_gradients([(grads, init_image)])
            clipped = tf.clip_by_value(init_image, min_vals, max_vals)
            init_image.assign(clipped)

            # TODO save iteration loss and process time
            logger.info(
                f'Iteration {i}\n{ loss = }\n{ style_score = }'
                f'\n{ content_score = }\n{ time.time()-run_start = }\n'
                f'{ time.time()-start = }'
                )
            
            if loss < best_loss:
                # Update best loss and best image from total loss. 
                best_loss = loss
                best_img = deprocess_img(init_image.numpy())
            
            # Save checkpoint image
            if i % checkpoint_interval == 0:
                save_img(best_img,run_id,i,output_folder)

    except Exception as e:
        logger.error(str(e), exc_info=e)
        save_img(best_img,run_id,i,output_folder,error=True)
        return False
    
    save_img(best_img,run_id,i+1,output_folder)
    return True