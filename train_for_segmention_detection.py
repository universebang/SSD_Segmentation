import numpy as np
import os
import csv
from data_generator.object_detection_2d_data_sort import Data_Input_Class
from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
import keras
from matplotlib import pyplot as plt
from skimage import io, measure

from models.keras_ssd7_new import build_model, init_Model
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

#from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_data_sort import Data_Input_Class
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.data_augmentation_chain_variable_input_size import DataAugmentationVariableInputSize
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation

########################################################################################################
img_height = 300 # Height of the input images
img_width = 480 # Width of the input images
img_channels = 3 # Number of color channels of the input images
intensity_mean = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
n_classes = 1 # Number of positive classes
#scales = [0.08, 0.16, 0.32, 0.64, 0.96] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
scales = [0.08, 0.16, 0.32, 0.64, 0.96, 1.05]
aspect_ratios = [1.0, 2.0, 0.5, 1.0/3.0] # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True # Whether or not the model is supposed to use coordinates relative to the image size
############################################################################################################

############################################################################################################
#根据mask生成目标外接标注框，并保存到csv文件
'''
img_dir = 'E:/0525/original'
mask_path = 'E:/0525/mask_richtig_4'
mask_images = os.listdir(mask_path)
print(mask_images)
mask_images.sort(key= lambda x:int(x[:-4]))
'''
'''
with open("C:/code/python/dataset_ssd/winterstreet/csv/test1.csv","w", newline = '') as csvfile:
    writer = csv.writer(csvfile)
    #先写入columns_name
    writer.writerow(["frame","xmin","xmax","ymin","ymax","class_id"])
    flag = True
    for i in range(len(mask_images)):
        image = io.imread(img_dir + '/' + mask_images[i])
        label_inter = measure.label(image)
        for region in measure.regionprops(label_inter):
            if flag:
                print(region.bbox)
                flag = False
'''

path1=os.path.abspath('.')   #表示当前所处的文件夹的绝对路径
path2=os.path.abspath('..')

trained_weight_path = os.path.join(path1, 'weightPlusVgg16', 'ssd7_epoch-01_loss-1.8788_val_loss-1.5055.h5')

############################################################################################################
def train(data_path):
    K.clear_session() # Clear previous models from memory.
    model = init_Model(trained_weight_path)

    img_dir = os.path.join(path2, 'dataset', data_path, 'original')
    labels_filename = os.path.join(path2, 'dataset', data_path, 'csv', 'labels_train.csv')
    mask_path =  os.path.join(path2, 'dataset', data_path, 'mask_richtig_4')
    data_input_class = Data_Input_Class(labels_filename = labels_filename, images_dir=img_dir, mask_groundTruth_Path= mask_path)

    images, filenames, labels, image_ids = data_input_class.parse_csv(ret = True)

    images = np.asarray(images)
    filenames = np.asarray(filenames)
    labels = np.asarray(labels)
    image_ids = np.asarray(image_ids)

    '''
    print('shape of images:', images.shape)
    print('shape of filenames:', filenames.shape)
    print('shape of labels:', labels.shape)
    print('shape of image_ids:', image_ids.shape)
    print(labels[1])
    print(image_ids)
    print(filenames[0:5])
    '''

    predictor_sizes = [
                    model.get_layer('classes3_3_add').output_shape[1:3],
                    model.get_layer('classes4').output_shape[1:3],
                    model.get_layer('classes5').output_shape[1:3],
                    model.get_layer('classes6').output_shape[1:3],
                    model.get_layer('classes7').output_shape[1:3]]


    ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                        img_width=img_width,
                                        n_classes=n_classes,
                                        predictor_sizes=predictor_sizes,
                                        scales=scales,
                                        aspect_ratios_global=aspect_ratios,
                                        two_boxes_for_ar1=two_boxes_for_ar1,
                                        steps=steps,
                                        offsets=offsets,
                                        clip_boxes=clip_boxes,
                                        variances=variances,
                                        matching_type='multi',
                                        pos_iou_threshold=0.5,
                                        neg_iou_limit=0.3,
                                        normalize_coords=normalize_coords)


    batch_X, batch_y_encoded = data_input_class.get_encoded_boxlabel(batch_size = 200, label_encoder=ssd_input_encoder)

    mask, class_weight = data_input_class.get_mask_label()

    '''
    print('type of batch_x:', type(batch_X))
    print('shape of batch_x:', batch_X.shape)
    print('type of batch_y:', type(batch_y_encoded))
    print('shape of batch_y_encoded:', batch_y_encoded.shape)
    '''
    print('shape of mask:', mask.shape)
    print('shape of class_weight:', class_weight.shape)


    #tb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, batch_size=batch_size, write_graph=False, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    #chk = keras.callbacks.ModelCheckpoint(mdl_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    #redu = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=reduce_factor, patience=num_patience, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    epoch = 50
    batch_size = 1

    #weight_saved_path = 'E:/0525/new_Version/ssd_keras-master-copy/weight_change/detection_segmentation/'
    weight_saved_path =  os.path.join(path2, 'dataset', data_path, 'weight', 'ssd_segmentation_detection.h5')
    chk = keras.callbacks.ModelCheckpoint(weight_saved_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    redu = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    model.fit(images, [batch_y_encoded, mask], validation_split=0.2, epochs=epoch, batch_size=batch_size, callbacks=[chk], verbose=1, class_weight=[None,class_weight], shuffle = True)

if __name__ == '__main__':

    #dic_dataset = {0: 'winterstreet', 1: 'highway'}
    dic_dataset = {0: 'cnitech_night_time'}
    for i in range(len(dic_dataset)):
        train(dic_dataset[i])
