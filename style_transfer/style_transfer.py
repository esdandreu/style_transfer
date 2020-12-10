import time
import logging

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path

from style_transfer.model import get_model, get_feature_representations
from style_transfer.loss import gram_matrix, compute_grads
from style_transfer.utils import load_and_process_img, deprocess_img, save_img
from style_transfer.config import (
    OUTPUT_FOLDER, CHECKPOINTS_PER_RUN
    )
from style_transfer._logging import stats_logger

import IPython.display

logger = logging.getLogger(__name__)

def run_style_transfer(
    content_path: Path, 
    style_path: Path,
    num_iterations: int = 1000,
    output_folder: Path = OUTPUT_FOLDER,
    content_weight: float = 1e3, 
    style_weight: float = 1e-2,
    verbose: bool = False,
    log_images: bool = False,
    ): 

    # Image and stats identifier form input parameters
    run_id = f'{content_path.stem}_{style_path.stem}'

    # We don't need to (or want to) train any layers of our model, so we set
    # their trainable to false. 

    # TODO get as input
    # Content layer where will pull our feature maps
    content_layers = ['block5_conv2'] 

    # Style layer we are interested in
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1'
                  ]

    model = get_model(
        content_layers=content_layers,
        style_layers=style_layers,
        )
    
    for layer in model.layers:
        layer.trainable = False
    
    # Argument for computing the gradients
    loss_weights = (style_weight, content_weight)
    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)
    
    # Get the style and content feature representations (from our specified
    # intermediate layers)
    style_features, content_features = get_feature_representations(
        model=model, 
        content_path=content_path, 
        style_path=style_path,
        num_content_layers=num_content_layers,
        num_style_layers=num_style_layers
        )
    gram_style_features = [
        gram_matrix(style_feature) for style_feature in style_features
        ]
    
    # Set initial image
    init_image = load_and_process_img(content_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)

    # Create our optimizer
    # TODO play with optimizer? 
    optimizer = tf.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)

    # Store our best result
    best_loss, best_img = float('inf'), None
    
        
    # ? What is this
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means   

    # Interval for checkpoints
    run_start = time.time()
    checkpoint_interval = num_iterations/(CHECKPOINTS_PER_RUN)

    # Create stats file
    stats = stats_logger(Path(output_folder, f'{run_id}.csv'))
    stats.info('run_time,iteration,loss,style_loss,content_loss,time')
    
    try: 
        for i in range(num_iterations):
            start = time.time()
            grads, loss, style_score, content_score = compute_grads(
                model=model,
                loss_weights=loss_weights,
                init_image=init_image,
                gram_style_features=gram_style_features,
                content_features=content_features,
                num_content_layers=num_content_layers,
                num_style_layers=num_style_layers
                )
            optimizer.apply_gradients([(grads, init_image)])
            clipped = tf.clip_by_value(init_image, min_vals, max_vals)
            init_image.assign(clipped)
            end = time.time()

            # Iteration stats
            stats_line = (
                f'{end-run_start:.4f},{i},{loss:.4e},{style_score:.4e},'
                f'{content_score:.4e},{end-start:.4f}'
                )

            
            # Update best loss and best image from total loss. 
            if loss < best_loss:
                best_loss = loss
                best_img = deprocess_img(init_image.numpy())
            
            # Save checkpoint image
            if i % checkpoint_interval == 0:
                plot_img = init_image.numpy()
                plot_img = deprocess_img(plot_img)
                save_img(best_img,run_id,i,output_folder)
                if log_images:
                    IPython.display.display_png(Image.fromarray(plot_img))
                stats.info(stats_line)
            elif not verbose:
                stats.debug(stats_line)
            else:
                stats.info(stats_line)

    except KeyboardInterrupt as e:
        logger.error('Keyboard Interrupt')
        save_img(best_img,run_id,i,output_folder,error=True)
        raise e
    except Exception as e:
        logger.error(str(e), exc_info=e)
        save_img(best_img,run_id,i,output_folder,error=True)
        return False
    finally:
        stats.handlers = []
    
    save_img(best_img,run_id,i+1,output_folder)
    return True