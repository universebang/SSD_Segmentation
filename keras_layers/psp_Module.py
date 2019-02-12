from math import ceil
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization, Activation, Input, Dropout, ZeroPadding2D, Lambda
from keras.layers.merge import Concatenate, Add
from keras.backend import tf as ktf
#from keras_layer_L2Normalization import L2Normalization


def BN(name=""):
    return BatchNormalization(momentum=0.95, name=name, epsilon=1e-5)

def Interp(x, shape):
    ''' 对图片做一个放缩，配合Keras的Lambda层使用'''
    new_height, new_width, depth = shape
    resized = ktf.image.resize_images(x, [new_height, new_width], align_corners=True)
    return resized

def interp_block(prev_layer, level, feature_map_shape):

    input_shape = feature_map_shape

    if input_shape == (300, 480, 64):
        kernel_strides_map = {1: 60,  # (300-60)/60 + 1 = 5
                              2: 30,  # (300-30)/30 + 1 = 10
                              3: 20,  # (300-20)/20 + 1 = 15
                              6: 10}  # (300-10)/10 + 1 = 30
        name_pre = "pyramide_1"
        
    elif input_shape == (150, 240, 128):
        kernel_strides_map = {1: 30,  # 5*8
                              2: 20,  # 7.5*12
                              3: 10,
                              5: 5}  # 15*24
        name_pre = "pyramide_2"
    elif input_shape == (75, 120, 256):
        kernel_strides_map = {1: 15,
                              2: 5}
        name_pre = "pyramide_3"
    elif input_shape == (75, 120, 512):
        kernel_strides_map = {1: 15,
                              2: 5}
        name_pre = "pyramide_4"
    else:
        print("Pooling parameters for input shape ", input_shape, " are not defined.")
        exit(1)

    names = [
        name_pre + str(level) + "_conv",
        name_pre + str(level) + "_conv_bn"
        ]
    kernel = (kernel_strides_map[level], kernel_strides_map[level])
    strides = (kernel_strides_map[level], kernel_strides_map[level])
    prev_layer = AveragePooling2D(kernel, strides=strides)(prev_layer) # 平均池化
    #prev_layer = Conv2D(int(input_shape[2]/len(kernel_strides_map)), (1, 1), strides=(1, 1), name=names[0], use_bias=False)(prev_layer) # 通道降到 原本的1/N = 1/4
    prev_layer = Conv2D(int(input_shape[2]/len(kernel_strides_map)), (1, 1), strides=(1, 1), name=names[0])(prev_layer)
    #prev_layer = BN(name=names[1])(prev_layer)
    prev_layer = Activation('relu')(prev_layer)
    prev_layer = Lambda(Interp, arguments={'shape': feature_map_shape})(prev_layer) # 放缩到指定大小
    return prev_layer

#################################################################################################
# feature map经过Pyramid Pooling Module得到融合的带有整体信息的feature，在上采样与池化前的feature map相concat
# 最后过一个卷积层得到最终输出
#################################################################################################

def build_pyramid_pooling_module(res, input_shape):
    """Build the Pyramid Pooling Module."""
    # ---PSPNet concat layers with Interpolation
    # feature_map_size, 池化的特征图将会线性插值放大到该size
    #feature_map_size = tuple(int(ceil(input_dim / 8.0)) for input_dim in input_shape)
    feature_map_size = input_shape
    print("PSP module will interpolate to a final feature map size of %s" % (feature_map_size, ))

    if input_shape == (300, 480, 64):
        level = [1, 2, 3, 6]
    elif input_shape == (150, 240, 128):
        level = [1, 2, 3, 5]
    elif input_shape == (75, 120, 256):
        level = [1, 2]
    elif input_shape == (75, 120, 512):
        level = [1, 2]
    else:
        print("Pooling parameters for input shape ", input_shape, " are not defined.")
        exit(1)

    list_concat = [res]
    for i in range(len(level)):
        re_interp_block = interp_block(res, level[i], feature_map_size)
        list_concat.append(re_interp_block)
    
    '''
    # 创建不同尺度的feature
    interp_block1 = interp_block(res, 1, feature_map_size)
    interp_block2 = interp_block(res, 2, feature_map_size)
    interp_block3 = interp_block(res, 3, feature_map_size)
    interp_block6 = interp_block(res, 6, feature_map_size)
    '''
    # 通道融合，融合所有feature 原本通道为2048  每层池化占512个通道  
    # shape=(1,feature_map_size_x,feature_map_size_y,4096)  融合后共4096个
    '''
    res = Concatenate()([res,
                         interp_block6,
                         interp_block3,
                         interp_block2,
                         interp_block1])
    '''
    res = Concatenate()(list_concat)
    return res

"""

"""