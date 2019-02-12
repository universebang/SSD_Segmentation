from keras.layers import *
import tensorflow as tf

# Refer to SE-Inception module (figure 2) and 
#          SE-ResNet module (figure 3) in the arxiv paper
# Input: input_tensor is the output tensor of residual/inception blocks
# Output: returns scaled input

def Squeeze_excitation_layer(input_tensor, compress_rate = 16, layer_name = None):
    '''
    input_tensor: input \n
    compress_rate = 16 (in default) \n
    layer_name = None (in default)
    '''
    with tf.name_scope(layer_name) :
        
        num_channels = int(input_tensor.shape[-1]) # Tensorflow backend
        bottle_neck = int(num_channels//compress_rate)
    
        se_branch = GlobalAveragePooling2D()(input_tensor)

        se_branch = Dense(bottle_neck, activation='relu', name = layer_name + '_fully_connected1')(se_branch)
        se_branch = Dense(num_channels, activation='sigmoid', name = layer_name + '_fully_connected2')(se_branch)

        x =  input_tensor
        out = multiply([x, se_branch])
    
        return out