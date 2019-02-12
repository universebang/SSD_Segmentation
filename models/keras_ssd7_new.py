'''
A small 7-layer Keras model with SSD architecture. Also serves as a template to build arbitrary network architectures.

Copyright (C) 2018 Pierluigi Ferrari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
from __future__ import division

import sys
sys.path.append("..")

import numpy as np
import keras
from keras import losses
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, Activation, Dropout
from keras.layers.convolutional import Conv2DTranspose
from keras.regularizers import l2
import keras.backend as K
from keras import regularizers
from keras.optimizers import Adam
from keras.layers.merge import Concatenate
from keras.backend import tf as ktf

from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_L2Normalization import L2Normalization
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.se_unit import Squeeze_excitation_layer
from keras_layers.psp_Module import *

reg = 5e-4
#def transposedConv(self, x, reuse_feature1, reuse_feature2, reuse_feature3, reuse_feature4):
'''
def transposedConv(x):
        
        # 实验结果： 利用SE_unit不利于semantic segmentation
        # 估计原因： SE_unit会压制一些Feature， 或许对classification有益， 但对于segmentation会起到想反作用
        # 有没有可能改进 SE_unit, 放大那些次要 Feature
        #reuse_feature1 = Squeeze_excitation_layer(reuse_feature1, compress_rate = 16, layer_name = 'se_layer1' )
        #resuse_feature2 = Squeeze_excitation_layer(resuse_feature2, compress_rate = 16, layer_name = 'se_layer2' )
        #resuse_feature3 = Squeeze_excitation_layer(resuse_feature3, compress_rate = 16, layer_name = 'se_layer3' ) 
        
        #reuse_feature1 = Conv2D(8, (1, 1), activation='relu', padding='same', name='block_conv1')(reuse_feature1)
        #reuse_feature2 = Conv2D(16, (1, 1), activation='relu', padding='same', name='block_conv2')(reuse_feature2)
        #reuse_feature3 = Conv2D(32, (1, 1), activation='relu', padding='same', name='block_conv3')(reuse_feature3)
        
        #reuse_feature1_con = Conv2D(16, (3, 3), strides = (2,2), activation='relu', padding='same', name='reuse_block_conv')(reuse_feature1)
        #reuse_feature2_1 = keras.layers.concatenate([reuse_feature2, reuse_feature1_con], name='resuse_concat1')
        #reuse_feature2_con = Conv2D(16, (3, 3), strides = (2,2), activation='relu', padding='same', name='reuse2_block_conv')(reuse_feature2_1)
        #resuse_feature3_2 = keras.layers.concatenate([reuse_feature3, reuse_feature2_con])

        #reuse_feature3 = Conv2D(32, (1, 1), activation='relu', padding='same', name='block_conv3')(reuse_feature4)
        # block 5
        x = Conv2DTranspose(64, (1, 1), activation='relu', padding='same', name='block5_tconv1', 
                                                kernel_regularizer=regularizers.l2(reg))(x)

        #x =  keras.layers.concatenate([x, reuse_feature4], name='feature_concat_reuse0')

        x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', name='block5_tconv2')(x)
        x = Conv2DTranspose(512, (1, 1), activation='relu', padding='same', name='block5_tconv3')(x)
        
        # block 6
        x = Conv2DTranspose(64, (1, 1), activation='relu', padding='same', name='block6_tconv1', 
                                                kernel_regularizer=regularizers.l2(reg))(x)

        #x =  keras.layers.concatenate([x, reuse_feature3], name='feature_concat_reuse1')

        x = Conv2DTranspose(64, (5, 5), strides=(2, 2), activation='relu', padding='same', name='block6_tconv2')(x)
        x = Conv2DTranspose(256, (1, 1), activation='relu', padding='same', name='block6_tconv3')(x)
        
        # block 7
        x = Conv2DTranspose(64, (1, 1), activation='relu', padding='same', name='block7_tconv1', 
                                                kernel_regularizer=regularizers.l2(reg))(x)
        x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', name='block7_tconv2')(x)
        x = Conv2DTranspose(128, (1, 1), activation='relu', padding='same', name='block7_tconv3')(x)

        #x =  keras.layers.concatenate([x, reuse_feature2], name='feature_concat_reuse2')
        
        # block 8
        x = Conv2DTranspose(64, (5, 5), strides=(2, 2), activation='relu', padding='same', name='block8_conv1', 
                                                kernel_regularizer=regularizers.l2(reg))(x)
        
        #x =  keras.layers.concatenate([x, reuse_feature1], name='feature_concat_reuse3')
        # block 9
        x = Conv2DTranspose(1, (1, 1), padding='same', name='block9_conv1')(x)
        x = Activation('sigmoid')(x)
        
        return x
'''
def feature_merge(forward_feature, reuse_feature, layer_name):

    reuse_feature = Conv2DTranspose(64, (1, 1), activation='relu', padding='same', name= layer_name + 'reuse', kernel_regularizer=regularizers.l2(reg))(reuse_feature)
    
    reuse_feature = L2Normalization(gamma_init=20, name=layer_name + 'reuse1')(reuse_feature)    
    forward_feature = L2Normalization(gamma_init=20, name=layer_name + 'forward')(forward_feature)

    forward_feature =  keras.layers.concatenate([forward_feature, reuse_feature], name= layer_name + 'feature_concat')
    #forward_feature = Conv2DTranspose(64, (1, 1), activation='relu', padding='same', name= layer_name + 'reuse1', kernel_regularizer=regularizers.l2(reg))(forward_feature)
    return forward_feature

def concat_local_global_feature(local_feature, pre_feature, input_shape, layer_name):
    '''
    Function: \n
            concat the feature in deconvolutional layer and feature of previous layer
    '''
    pre_feature_new = build_pyramid_pooling_module(pre_feature, input_shape)
    #pre_feature_new = Conv2DTranspose(64, (1, 1), activation='relu', padding='same', name= layer_name + 'tconv')(pre_feature_new)
    pre_feature_new = L2Normalization(gamma_init=20, name= layer_name + 'reuse_feature1_new_norm')(pre_feature_new)
    local_feature = L2Normalization(gamma_init=20, name= layer_name + 'x_norm')(local_feature)
    concat = Concatenate()([local_feature, pre_feature_new])

    return concat

def transposedConv(x_input, reuse_feature1, reuse_feature2, reuse_feature3, reuse_feature4):
        # block 5
        '''
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        #x = Dropout(0.5, name='dr1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        #x = Dropout(0.5, name='dr2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        #x = Dropout(0.5, name='dr3')(x)
        '''
        #se_output = Conv2DTranspose(64, (1, 1), activation='relu', padding='same', name='se_output1', kernel_regularizer=regularizers.l2(reg))(x_input)
        #se_output = Squeeze_excitation_layer(se_output, layer_name= 'se_layer1')
        #se_output = Conv2DTranspose(64, (1, 1), activation='relu', padding='same', name='se_output1', kernel_regularizer=regularizers.l2(reg))(se_output)
        '''
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x_input)
        x = Dropout(0.5, name='dr1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Dropout(0.5, name='dr2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = Dropout(0.5, name='dr3')(x)
        '''
        #x = Dropout(0.5, name='dr3')(x_input)
        '''
        x = Conv2DTranspose(64, (1, 1), activation='relu', padding='same', name='block5_tconv1', kernel_regularizer=regularizers.l2(reg))(x_input) 

        #x = feature_merge(x, reuse_feature4, layer_name = '1')
        
        x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', name='block5_tconv2')(x)
        x = Conv2DTranspose(512, (1, 1), activation='relu', padding='same', name='block5_tconv3')(x)
        
        # block 6
        x = Conv2DTranspose(64, (1, 1), activation='relu', padding='same', name='block6_tconv1', kernel_regularizer=regularizers.l2(reg))(x)
        #新加##############################################
        #x = feature_merge(x, reuse_feature3, layer_name = '2')
        #x =  keras.layers.concatenate([x, reuse_feature3], name='feature_concat_reuse3')
        
        ##################################################
        x = Conv2DTranspose(64, (5, 5), strides=(2, 2), activation='relu', padding='same', name='block6_tconv2')(x)
        x = Conv2DTranspose(256, (1, 1), activation='relu', padding='same', name='block6_tconv3')(x)
        
        # block 7
        x = Conv2DTranspose(64, (1, 1), activation='relu', padding='same', name='block7_tconv1', kernel_regularizer=regularizers.l2(reg))(x)
        #新加##############################################
        #x = concat_local_global_feature(x, reuse_feature2, (150, 240, 64), layer_name = 'concat2')
        ##################################################
        x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', name='block7_tconv2')(x)
        x = Conv2DTranspose(128, (1, 1), activation='relu', padding='same', name='block7_tconv3')(x)        
        
        # block 8
        x = Conv2DTranspose(64, (5, 5), strides=(2, 2), activation='relu', padding='same', name='block8_conv1', kernel_regularizer=regularizers.l2(reg))(x)        
        #x = Conv2DTranspose(128, (1, 1), activation='relu', padding='same', name='block7_tconv3')(x)
        '''
        #x = concat_local_global_feature(x_input, reuse_feature4, (75, 120, 512), layer_name = 'concat1')
        x = Conv2DTranspose(64, (1, 1), activation='relu', padding='same', name='1')(x_input)
        x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', name='block7_tconv3')(x)
        x = Conv2DTranspose(256, (1, 1), activation='relu', padding='same', name='2')(x)
        
        #x = concat_local_global_feature(x, reuse_feature3, (75, 120, 256), layer_name = '3')
        
        
        x = Conv2DTranspose(64, (1, 1), activation='relu', padding='same', name='4')(x)
        x = Conv2DTranspose(64, (5, 5), strides=(2, 2), activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg), name='4_1')(x)

        x = Conv2DTranspose(128, (1, 1), activation='relu', padding='same', name='5')(x)
        x = concat_local_global_feature(x, reuse_feature2, (150, 240, 128), layer_name = '6')
        x = Conv2DTranspose(64, (1, 1), activation='relu', padding='same', name='7')(x)

        x = Conv2DTranspose(64, (5, 5), strides=(2, 2), activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg), name='block8_conv1')(x)
        x = concat_local_global_feature(x, reuse_feature1, (300, 480, 64), layer_name = '8')
        
        # block 9
        x = Conv2DTranspose(1, (1, 1), padding='same', name='block9_conv1')(x)

        x = Activation('sigmoid')(x)
        

        return x

def build_model(image_size,
                n_classes,
                mode='training',
                l2_regularization=0.0,
                min_scale=0.1,
                max_scale=0.9,
                scales=None,
                aspect_ratios_global=[0.5, 1.0, 2.0],
                aspect_ratios_per_layer=None,
                two_boxes_for_ar1=True,
                steps=None,
                offsets=None,
                clip_boxes=False,
                variances=[1.0, 1.0, 1.0, 1.0],
                coords='centroids',
                normalize_coords=False,
                subtract_mean=None,
                divide_by_stddev=None,
                swap_channels=False,
                confidence_thresh=0.01,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400,
                return_predictor_sizes=False):
    '''
    Build a Keras model with SSD architecture, see references.

    The model consists of convolutional feature layers and a number of convolutional
    predictor layers that take their input from different feature layers.
    The model is fully convolutional.

    The implementation found here is a smaller version of the original architecture
    used in the paper (where the base network consists of a modified VGG-16 extended
    by a few convolutional feature layers), but of course it could easily be changed to
    an arbitrarily large SSD architecture by following the general design pattern used here.
    This implementation has 7 convolutional layers and 4 convolutional predictor
    layers that take their input from layers 4, 5, 6, and 7, respectively.

    Most of the arguments that this function takes are only needed for the anchor
    box layers. In case you're training the network, the parameters passed here must
    be the same as the ones used to set up `SSDBoxEncoder`. In case you're loading
    trained weights, the parameters passed here must be the same as the ones used
    to produce the trained weights.

    Some of these arguments are explained in more detail in the documentation of the
    `SSDBoxEncoder` class.

    Note: Requires Keras v2.0 or later. Training currently works only with the
    TensorFlow backend (v1.0 or later).

    Arguments:
        image_size (tuple): The input image size in the format `(height, width, channels)`.
        n_classes (int): The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO.
        mode (str, optional): One of 'training', 'inference' and 'inference_fast'. In 'training' mode,
            the model outputs the raw prediction tensor, while in 'inference' and 'inference_fast' modes,
            the raw predictions are decoded into absolute coordinates and filtered via confidence thresholding,
            non-maximum suppression, and top-k filtering. The difference between latter two modes is that
            'inference' follows the exact procedure of the original Caffe implementation, while
            'inference_fast' uses a faster prediction decoding procedure.
        l2_regularization (float, optional): The L2-regularization rate. Applies to all convolutional layers.
        min_scale (float, optional): The smallest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images.
        max_scale (float, optional): The largest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images. All scaling factors between the smallest and the
            largest will be linearly interpolated. Note that the second to last of the linearly interpolated
            scaling factors will actually be the scaling factor for the last predictor layer, while the last
            scaling factor is used for the second box for aspect ratio 1 in the last predictor layer
            if `two_boxes_for_ar1` is `True`.
        scales (list, optional): A list of floats containing scaling factors per convolutional predictor layer.
            This list must be one element longer than the number of predictor layers. The first `k` elements are the
            scaling factors for the `k` predictor layers, while the last element is used for the second box
            for aspect ratio 1 in the last predictor layer if `two_boxes_for_ar1` is `True`. This additional
            last scaling factor must be passed either way, even if it is not being used. If a list is passed,
            this argument overrides `min_scale` and `max_scale`. All scaling factors must be greater than zero.
        aspect_ratios_global (list, optional): The list of aspect ratios for which anchor boxes are to be
            generated. This list is valid for all predictor layers. The original implementation uses more aspect ratios
            for some predictor layers and fewer for others. If you want to do that, too, then use the next argument instead.
        aspect_ratios_per_layer (list, optional): A list containing one aspect ratio list for each predictor layer.
            This allows you to set the aspect ratios for each predictor layer individually. If a list is passed,
            it overrides `aspect_ratios_global`.
        two_boxes_for_ar1 (bool, optional): Only relevant for aspect ratio lists that contain 1. Will be ignored otherwise.
            If `True`, two anchor boxes will be generated for aspect ratio 1. The first will be generated
            using the scaling factor for the respective layer, the second one will be generated using
            geometric mean of said scaling factor and next bigger scaling factor.
        steps (list, optional): `None` or a list with as many elements as there are predictor layers. The elements can be
            either ints/floats or tuples of two ints/floats. These numbers represent for each predictor layer how many
            pixels apart the anchor box center points should be vertically and horizontally along the spatial grid over
            the image. If the list contains ints/floats, then that value will be used for both spatial dimensions.
            If the list contains tuples of two ints/floats, then they represent `(step_height, step_width)`.
            If no steps are provided, then they will be computed such that the anchor box center points will form an
            equidistant grid within the image dimensions.
        offsets (list, optional): `None` or a list with as many elements as there are predictor layers. The elements can be
            either floats or tuples of two floats. These numbers represent for each predictor layer how many
            pixels from the top and left boarders of the image the top-most and left-most anchor box center points should be
            as a fraction of `steps`. The last bit is important: The offsets are not absolute pixel values, but fractions
            of the step size specified in the `steps` argument. If the list contains floats, then that value will
            be used for both spatial dimensions. If the list contains tuples of two floats, then they represent
            `(vertical_offset, horizontal_offset)`. If no offsets are provided, then they will default to 0.5 of the step size,
            which is also the recommended setting.
        clip_boxes (bool, optional): If `True`, clips the anchor box coordinates to stay within image boundaries.
        variances (list, optional): A list of 4 floats >0. The anchor box offset for each coordinate will be divided by
            its respective variance value.
        coords (str, optional): The box coordinate format to be used internally by the model (i.e. this is not the input format
            of the ground truth labels). Can be either 'centroids' for the format `(cx, cy, w, h)` (box center coordinates, width,
            and height), 'minmax' for the format `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
        normalize_coords (bool, optional): Set to `True` if the model is supposed to use relative instead of absolute coordinates,
            i.e. if the model predicts box coordinates within [0,1] instead of absolute coordinates.
        subtract_mean (array-like, optional): `None` or an array-like object of integers or floating point values
            of any shape that is broadcast-compatible with the image shape. The elements of this array will be
            subtracted from the image pixel intensity values. For example, pass a list of three integers
            to perform per-channel mean normalization for color images.
        divide_by_stddev (array-like, optional): `None` or an array-like object of non-zero integers or
            floating point values of any shape that is broadcast-compatible with the image shape. The image pixel
            intensity values will be divided by the elements of this array. For example, pass a list
            of three integers to perform per-channel standard deviation normalization for color images.
        swap_channels (list, optional): Either `False` or a list of integers representing the desired order in which the input
            image channels should be swapped.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
            positive class in order to be considered for the non-maximum suppression stage for the respective class.
            A lower value will result in a larger part of the selection process being done by the non-maximum suppression
            stage, while a larger value will result in a larger part of the selection process happening in the confidence
            thresholding stage.
        iou_threshold (float, optional): A float in [0,1]. All boxes that have a Jaccard similarity of greater than `iou_threshold`
            with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
            to the box's confidence score.
        top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
            non-maximum suppression stage.
        nms_max_output_size (int, optional): The maximal number of predictions that will be left over after the NMS stage.
        return_predictor_sizes (bool, optional): If `True`, this function not only returns the model, but also
            a list containing the spatial dimensions of the predictor layers. This isn't strictly necessary since
            you can always get their sizes easily via the Keras API, but it's convenient and less error-prone
            to get them this way. They are only relevant for training anyway (SSDBoxEncoder needs to know the
            spatial dimensions of the predictor layers), for inference you don't need them.

    Returns:
        model: The Keras SSD model.
        predictor_sizes (optional): A Numpy array containing the `(height, width)` portion
            of the output tensor shape for each convolutional predictor layer. During
            training, the generator function needs this in order to transform
            the ground truth labels into tensors of identical structure as the
            output tensors of the model, which is in turn needed for the cost
            function.

    References:
        https://arxiv.org/abs/1512.02325v5
    '''

    n_predictor_layers = 5 # The number of predictor conv layers in the network
    n_classes += 1 # Account for the background class.
    l2_reg = l2_regularization # Make the internal name shorter.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################

    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError("`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers+1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(n_predictor_layers+1, len(scales)))
    else: # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)

    if len(variances) != 4: # We need one variance value for each of the four box coordinates
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1) # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    else: # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    ############################################################################
    # Define functions for the Lambda layers below.
    ############################################################################

    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]], tensor[...,swap_channels[3]]], axis=-1)

    ############################################################################
    # Build the network.
    ############################################################################

    x = Input(shape=(img_height, img_width, img_channels))
    '''
    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)
    if not (subtract_mean is None):#
        x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels), name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels), name='input_stddev_normalization')(x1)
    if swap_channels:
        x1 = Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(x1)
    '''
    '''
    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1_1')(x1)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1_2')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(conv1_2)


    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2_1')(pool1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(conv2_2)

    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_1')(pool2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_3')(conv3_2)
    #pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(conv3_3)

    # Block additional
    conv3_1_add = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),  name='conv3_1_add')(conv3_3)
    conv3_1_add = Dropout(0.5, name='dr1')(conv3_1_add)
    conv3_2_add = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_2_add')(conv3_1_add)
    conv3_2_add = Dropout(0.5, name='dr2')(conv3_2_add)
    conv3_3_add = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_3_add')(conv3_2_add)
    conv3_3_add = Dropout(0.5, name='dr3')(conv3_3_add)
    '''
    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_last')(x)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(conv1_1)
    reuse_feature1 = conv1_2
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv1_2)
    
    # Block 2
    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(pool1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(conv2_1)
    reuse_feature2 = conv2_2
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv2_2)
    
    # Block 3
    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(pool2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(conv3_2)
    reuse_feature3 = conv3_3
    
    # Block 4
    ###########################################################################################################################################
    conv3_1_add = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal',  kernel_regularizer=l2(l2_reg), name='block4_conv1')(conv3_3)
    conv3_1_add = ELU(name='elu3_1')(conv3_1_add)

    conv3_2_add = Conv2D(512, (3, 3), padding='same',kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),name='block4_conv2')(conv3_1_add)
    conv3_2_add = ELU(name='elu3_2')(conv3_2_add)

    conv3_3_add = Conv2D(512, (3, 3),padding='same', kernel_initializer='he_normal',  kernel_regularizer=l2(l2_reg),name='block4_conv3')(conv3_2_add)
    conv3_3_add = ELU(name='elu3_3')(conv3_3_add)

    print('shape of conv3_3:', conv3_3_add.get_shape().as_list())

    reuse_feature4 = conv3_3_add
    
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(conv3_3_add)
    ###########################################################################################################################################

    conv4 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4')(conv3_3_add)
    conv4 = BatchNormalization(axis=3, momentum=0.99, name='bn4')(conv4)
    conv4 = ELU(name='elu4')(conv4)
    print('shape of conv4:', conv4.get_shape().as_list())
    pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4)

    conv5 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5')(pool4)
    conv5 = BatchNormalization(axis=3, momentum=0.99, name='bn5')(conv5)
    conv5 = ELU(name='elu5')(conv5)
    print('shape of conv5:', conv5.get_shape().as_list())
    pool5 = MaxPooling2D(pool_size=(2, 2), name='pool5')(conv5)

    conv6 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6')(pool5)
    conv6 = BatchNormalization(axis=3, momentum=0.99, name='bn6')(conv6)
    conv6 = ELU(name='elu6')(conv6)
    print('shape of conv6:', conv6.get_shape().as_list())
    pool6 = MaxPooling2D(pool_size=(2, 2), name='pool6')(conv6)

    conv7 = Conv2D(32, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7')(pool6)
    conv7 = BatchNormalization(axis=3, momentum=0.99, name='bn7')(conv7)
    conv7 = ELU(name='elu7')(conv7)
    print('shape of conv7:', conv7.get_shape().as_list())

    '''
    shape of conv3_3: [None, 75, 120, 512]
    shape of conv4: [None, 75, 120, 64]
    shape of conv5: [None, 37, 60, 48]
    shape of conv6: [None, 18, 30, 48]
    shape of conv7: [None, 9, 15, 32]
    '''

    # Feed conv4_3 into the L2 normalization layer
    #conv3_3_norm = L2Normalization(gamma_init=20, name='conv3_3_norm')(conv3_3)
    conv3_3_add_norm = L2Normalization(gamma_init=20, name='conv3_3_add_norm')(conv3_3_add)
    # The next part is to add the convolutional predictor layers on top of the base network
    # that we defined above. Note that I use the term "base network" differently than the paper does.
    # To me, the base network is everything that is not convolutional predictor layers or anchor
    # box layers. In this case we'll have four predictor layers, but of course conv3_3_add
    # easily rewrite this into an arbitrarily deep base network and add an arbitrary number of
    # predictor layers on top of the base network by simply following the pattern shown here.

    # Build the convolutional predictor layers on top of conv layers 4, 5, 6, and 7.
    # We build two predictor layers on top of each of these layers: One for class prediction (classification), one for box coordinate prediction (localization)
    # We precidt `n_classes` confidence values for each box, hence the `classes` predictors have depth `n_boxes * n_classes`
    # We predict 4 box coordinates for each box, hence the `boxes` predictors have depth `n_boxes * 4`
    # Output shape of `classes`: `(batch, height, width, n_boxes * n_classes)`
    #classes3_3 = Conv2D(n_boxes[0] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes3_3')(conv3_3_norm)
    classes3_3_add = Conv2D(n_boxes[0] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes3_3_add')(conv3_3_add_norm)
    classes4 = Conv2D(n_boxes[1] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes4')(conv4)
    classes5 = Conv2D(n_boxes[2] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes5')(conv5)
    classes6 = Conv2D(n_boxes[3] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes6')(conv6)
    classes7 = Conv2D(n_boxes[4] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes7')(conv7)
    # Output shape of `boxes`: `(batch, height, width, n_boxes * 4)`
    #boxes3_3 = Conv2D(n_boxes[0] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes3_3')(conv3_3_norm)
    boxes3_3_add = Conv2D(n_boxes[0] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes3_3_add')(conv3_3_add_norm)
    boxes4 = Conv2D(n_boxes[1] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes4')(conv4)
    boxes5 = Conv2D(n_boxes[2] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes5')(conv5)
    boxes6 = Conv2D(n_boxes[3] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes6')(conv6)
    boxes7 = Conv2D(n_boxes[4] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes7')(conv7)

    # Generate the anchor boxes
    # Output shape of `anchors`: `(batch, height, width, n_boxes, 8)`  
    '''
    anchors3_3 = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0],
                                             two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0], clip_boxes=clip_boxes,
                                             variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors3_3')(boxes3_3)
    '''
    anchors3_3_add = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0],
                                    two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0], clip_boxes=clip_boxes,
                                    variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors3_3_add')(boxes3_3_add)
    anchors4 = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors4')(boxes4)
    anchors5 = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2], this_offsets=offsets[2], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors5')(boxes5)
    anchors6 = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios[3],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3], this_offsets=offsets[3], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors6')(boxes6)
    anchors7 = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5], aspect_ratios=aspect_ratios[4],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4], this_offsets=offsets[4], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors7')(boxes7)

    '''
    anchors4 = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors4')(boxes4)
    anchors5 = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors5')(boxes5)
    anchors6 = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2], this_offsets=offsets[2],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors6')(boxes6)
    anchors7 = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios[3],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3], this_offsets=offsets[3],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors7')(boxes7)
    '''
    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    #classes3_3_reshaped = Reshape((-1, n_classes), name='classes3_3_reshape')(classes3_3)
    classes3_3_add_reshaped = Reshape((-1, n_classes), name='classes3_3_add_reshape')(classes3_3_add)
    classes4_reshaped = Reshape((-1, n_classes), name='classes4_reshape')(classes4)
    classes5_reshaped = Reshape((-1, n_classes), name='classes5_reshape')(classes5)
    classes6_reshaped = Reshape((-1, n_classes), name='classes6_reshape')(classes6)
    classes7_reshaped = Reshape((-1, n_classes), name='classes7_reshape')(classes7)
    # Reshape the box coordinate predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    #boxes3_3_reshaped = Reshape((-1, 4), name='boxes3_3_reshape')(boxes3_3)
    boxes3_3_add_reshaped = Reshape((-1, 4), name='boxes3_3_add_reshape')(boxes3_3_add)
    boxes4_reshaped = Reshape((-1, 4), name='boxes4_reshape')(boxes4)
    boxes5_reshaped = Reshape((-1, 4), name='boxes5_reshape')(boxes5)
    boxes6_reshaped = Reshape((-1, 4), name='boxes6_reshape')(boxes6)
    boxes7_reshaped = Reshape((-1, 4), name='boxes7_reshape')(boxes7)
    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    #anchors3_3_reshaped = Reshape((-1, 8), name='anchors3_3_reshape')(anchors3_3)
    anchors3_3_add_reshaped = Reshape((-1, 8), name='anchors3_3_add_reshape')(anchors3_3_add)
    anchors4_reshaped = Reshape((-1, 8), name='anchors4_reshape')(anchors4)
    anchors5_reshaped = Reshape((-1, 8), name='anchors5_reshape')(anchors5)
    anchors6_reshaped = Reshape((-1, 8), name='anchors6_reshape')(anchors6)
    anchors7_reshaped = Reshape((-1, 8), name='anchors7_reshape')(anchors7)

    # Concatenate the predictions from the different layers and the assosciated anchor box tensors
    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1
    # Output shape of `classes_concat`: (batch, n_boxes_total, n_classes)
    classes_concat = Concatenate(axis=1, name='classes_concat')([classes3_3_add_reshaped,
                                                                 classes4_reshaped,
                                                                 classes5_reshaped,
                                                                 classes6_reshaped,
                                                                 classes7_reshaped])

    # Output shape of `boxes_concat`: (batch, n_boxes_total, 4)
    boxes_concat = Concatenate(axis=1, name='boxes_concat')([boxes3_3_add_reshaped,
                                                             boxes4_reshaped,
                                                             boxes5_reshaped,
                                                             boxes6_reshaped,
                                                             boxes7_reshaped])

    # Output shape of `anchors_concat`: (batch, n_boxes_total, 8)
    anchors_concat = Concatenate(axis=1, name='anchors_concat')([anchors3_3_add_reshaped,
                                                                 anchors4_reshaped,
                                                                 anchors5_reshaped,
                                                                 anchors6_reshaped,
                                                                 anchors7_reshaped])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    classes_softmax = Activation('softmax', name='classes_softmax')(classes_concat)

    # Concatenate the class and box coordinate predictions and the anchors to one large predictions tensor
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([classes_softmax, boxes_concat, anchors_concat])

    print('shape of class_softmax:', classes_softmax.shape)  #(?, 119580, 6)
    print('shape of boxes_concat:', boxes_concat.shape)      #(?, 119580, 4)
    print('shape of anchors_concat:', anchors_concat.shape)  #(?, 119580, 8)
    print('shape of predictions:', predictions.shape)        #(?, 119580, 18)

    #generated_mask = transposedConv(conv3_3_add)
    generated_mask = transposedConv(conv3_3_add, reuse_feature1, reuse_feature2, reuse_feature3, reuse_feature4)

    if mode == 'training':
        model = Model(inputs=x, outputs=[predictions, generated_mask])
        #model = Model(inputs=x, outputs=[predictions])
        #model = Model(inputs=x, outputs=[generated_mask])
    elif mode == 'inference':
        decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               coords=coords,
                                               normalize_coords=normalize_coords,
                                               img_height=img_height,
                                               img_width=img_width,
                                               name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    elif mode == 'inference_fast':
        decoded_predictions = DecodeDetectionsFast(confidence_thresh=confidence_thresh,
                                                   iou_threshold=iou_threshold,
                                                   top_k=top_k,
                                                   nms_max_output_size=nms_max_output_size,
                                                   coords=coords,
                                                   normalize_coords=normalize_coords,
                                                   img_height=img_height,
                                                   img_width=img_width,
                                                   name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    else:
        raise ValueError("`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(mode))

    if return_predictor_sizes:
        # The spatial dimensions are the same for the `classes` and `boxes` predictor layers.
        predictor_sizes = np.array([classes3_3._keras_shape[1:3],
                                    classes3_3_add._keras_shape[1:3],
                                    classes4._keras_shape[1:3],
                                    classes5._keras_shape[1:3],
                                    classes6._keras_shape[1:3],
                                    classes7._keras_shape[1:3]])
        return model, predictor_sizes
    else:
        return model

def init_Model(pre_trained_weight):
    ##################################################################################################################
    img_height = 300 # Height of the input images
    img_width = 480 # Width of the input images
    img_channels = 3 # Number of color channels of the input images
    intensity_mean = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
    intensity_range = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
    n_classes = 1 # Number of positive classes
    #scales = [0.08, 0.16, 0.32, 0.64, 0.96, 1] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
    scales = [0.08, 0.16, 0.32, 0.64, 0.96, 1.05]
    aspect_ratios = [1.0, 2.0, 0.5, 1.0/3.0] # The list of aspect ratios for the anchor boxes
    two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
    steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
    offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
    clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
    variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
    normalize_coords = True # Whether or not the model is supposed to use coordinates relative to the image size
    ###################################################################################################################

    model = build_model(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='training',
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_global=aspect_ratios,
                    aspect_ratios_per_layer=None,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=intensity_mean,
                    divide_by_stddev=intensity_range)

    # 2: Optional: Load some weights

    #model.load_weights(r'C:/code/python/ssd_keras-master/weight/ssd7_epoch-00_loss-2.1119_val_loss-1.7316.h5', by_name=True)
    #model.load_weights('/home/myubuntu/new_Version/ssd_keras-master-copy/weightPlusVgg16/ssd7_epoch-01_loss-2.2647_val_loss-1.7594.h5', by_name=True)
    #model.load_weights('/home/myubuntu/FgSegNet-master/FgSegNet-master/FgSegNet/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)
    model.load_weights(pre_trained_weight, by_name=True)
    #model.load_weights('/home/myubuntu/new_Version/ssd_keras-master-copy/weight_change/segmentation/ssd7_segmentation.h5', by_name=True)

    # add: freeze some layer
    freeze_layers = ['identity_layer', 'input_mean_normalization', 'input_stddev_normalization', 'block1_conv1', 'block1_conv2', 'block1_pool',
                     'block2_conv1', 'block2_conv2', 'block2_pool', 'block3_conv1', 'block3_conv2', 'block3_conv3']

    
    for layer in model.layers:
        if layer.name in freeze_layers:
            layer.trainable = False
          
    # 3: Instantiate an Adam optimizer and the SSD loss function and compile the model


    # for detection
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

    # for segmentation
    opt = keras.optimizers.RMSprop(lr = 1e-4, rho=0.9, epsilon=1e-08, decay=0.0)
    #model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    #model.compile(optimizer=adam, loss='binary_crossentropy')
    #model.compile(optimizer=opt, loss=ssd_loss.compute_loss)
    model.compile(optimizer=opt, loss=[ssd_loss.compute_loss, losses.binary_crossentropy], loss_weights = [1, 10])

    return model

