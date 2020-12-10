# loss.py
import tensorflow as tf

def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))


def gram_matrix(input_tensor):
    # We make the image channels first
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def get_style_loss(base_style, gram_target):
    """Expects two images of dimension h, w, c"""
    # height, width, num filters of each layer

    # We scale the loss at a given layer by the size of the feature map and the
    # number of filters
    # TODO remove?
    # height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)

    return tf.reduce_mean(tf.square(gram_style - gram_target))# / (4. * (channels ** 2) * (width * height) ** 2)


def compute_loss(
    model: tf.keras.models.Model,
    loss_weights, 
    init_image, 
    gram_style_features, 
    content_features,
    num_content_layers,
    num_style_layers
    ):
    """This function will compute the loss total loss.

    Arguments: model: The model that will give us access to the intermediate
        layers loss_weights: The weights of each contribution of each loss
        function. (style weight, content weight, and total variation weight)
        init_image: Our initial base image. This image is what we are updating
        with our optimization process. We apply the gradients wrt the loss we
        are calculating to this image. gram_style_features: Precomputed gram
        matrices corresponding to the defined style layers of interest.
        content_features: Precomputed outputs from defined content layers of
        interest.

    Returns: returns the total loss, style loss, content loss, and total
        variational loss
    """
    style_weight, content_weight = loss_weights
    
    # Feed our init image through our model. This will give us the content and 
    # style representations at our desired layers. Since we're using eager
    # our model is callable just like any other function!
    model_outputs = model(init_image)
    
    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]
    
    style_score = 0
    content_score = 0

    # Accumulate losses from all layers
    # Here, we equally weight each contribution of each loss layer
    layer_weight = 1.0 / float(num_style_layers)
    for target, comb in zip(gram_style_features, style_output_features):
        style_score += layer_weight*get_style_loss(comb[0], target)
    layer_weight = 1.0 / float(num_content_layers)
    for target, comb in zip(content_features, content_output_features):
        content_score += layer_weight*get_content_loss(comb[0], target)
    
    style_score *= style_weight
    content_score *= content_weight

    # Get total loss
    loss = style_score + content_score 
    return loss, style_score, content_score

def compute_grads(
    model: tf.keras.models.Model,
    loss_weights, 
    init_image, 
    gram_style_features, 
    content_features,
    num_content_layers,
    num_style_layers
    ):
    with tf.GradientTape() as tape: #? Why tape
        loss, style_score, content_score = compute_loss(
            model=model,
            loss_weights=loss_weights, 
            init_image=init_image, 
            gram_style_features=gram_style_features,
            content_features=content_features,
            num_content_layers=num_content_layers,
            num_style_layers=num_style_layers
        )
        # Compute gradients wrt input image
        return (
            tape.gradient(loss, init_image),
            loss, style_score, content_score
            )