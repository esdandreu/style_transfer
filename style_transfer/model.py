from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19

from style_transfer.utils import load_and_process_img

def get_model(content_layers, style_layers):
    """ Creates our model with access to intermediate layers. 
    
    This function will load the VGG19 model and access the intermediate layers. 
    These layers will then be used to create a new model that will take input image
    and return the outputs from these intermediate layers from the VGG model. 
    
    Returns:
        returns a keras model that takes image inputs and outputs the style and 
        content intermediate layers. 
    """
    # TODO flexible pretrain and flexible layer number
    # Load our model. We load pretrained VGG, trained on imagenet data
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    # Get output layers corresponding to style and content layers 
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    # Build model
    model = Model(vgg.input, model_outputs)
    return model

def get_feature_representations(
    model,
    content_path, 
    style_path,
    num_content_layers,
    num_style_layers
    ):
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
                      for style_layer in style_outputs[:num_style_layers]
                      ]
    content_features = [
                        content_layer[0] 
                        for content_layer in content_outputs[num_style_layers:]
                        ]
    return style_features, content_features