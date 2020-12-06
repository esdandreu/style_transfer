from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19

# TODO build in function
# Content layer where will pull our feature maps
content_layers = ['block5_conv2'] 

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]

def get_model():
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
    model.style_layers = style_layers
    model.content_layers = content_layers
    model.num_style_layers = len(style_layers)
    model.num_content_layers = len(content_layers)
    return model