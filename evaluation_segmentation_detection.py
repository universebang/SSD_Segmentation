from math import ceil

import numpy as np
import os
import scipy
import cv2
import time
from keras import backend as K
from keras.callbacks import (CSVLogger, EarlyStopping, ModelCheckpoint,
                             ReduceLROnPlateau, TerminateOnNaN)
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing import image as kImage
from matplotlib import pyplot as plt
from skimage import io, transform
from keras.preprocessing import image as kImage

from draw_mask_on_image import draw_mask_on_image_array
from convert_frames_to_video import frames_to_video

from data_generator.data_augmentation_chain_constant_input_size import \
    DataAugmentationConstantInputSize
from data_generator.data_augmentation_chain_original_ssd import \
    SSDDataAugmentation
from data_generator.data_augmentation_chain_variable_input_size import \
    DataAugmentationVariableInputSize
from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_misc_utils import \
    apply_inverse_transforms
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_loss_function.keras_ssd_loss import SSDLoss
from models.keras_ssd7_new import build_model
from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import (decode_detections,
                                                    decode_detections_fast)

#################################################################################################################
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
##################################################################################################################

##################################################################################################################
# 1: Build the Keras model

def read_image(dir='', as_grey = False):
    imgs = os.listdir(dir)
    img_num = len(imgs)
    imgs.sort(key= lambda x:int(x[:-4])) 

    if as_grey:
        pass
    else:
        train_images_all = []
        for i in range(img_num):
            #im = kImage.load_img(dir+ '/' + imgs[i])
            #im = kImage.img_to_array(im)
            im = io.imread(dir+ '/' + imgs[i])
            train_images_all.append(im)
            #print('data_dir:', dir + '/' + imgs[i])
    return np.asarray(train_images_all)


def evaluation(file_name):

    K.clear_session() # Clear previous models from memory.

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
    path= os.path.abspath('..') 

    weight_saved_path = os.path.join(path, 'dataset', file_name, 'weight', 'ssd_segmentation_detection.h5')
    test_images_dir = os.path.join(path, 'dataset', file_name, 'test')
    mask_saved_dir = os.path.join(path, 'dataset', file_name, 'result')
    render_image_dir = os.path.join(path, 'dataset', file_name, 'rendering')
    demo_video_dir = os.path.join(path, 'dataset', file_name, 'demo', 'demo.avi')
    #original = os.path.join(path, 'dataset', file_name, 'original')

    model.load_weights(weight_saved_path, by_name=True)

    #######################################################################################################
    all_images = read_image(test_images_dir)
   
    #print('shape of batch_images:', batch_images.shape)

    
    #plt.figure(figsize=(20,12))
    plt.figure()
    plt.imshow(all_images[0])
    plt.show()

    img_num = all_images.shape[0]
    batch_size = 3
    batch_num = int(img_num/batch_size)
   
    id = 1
    
    start_time = time.time()

    for i in range(batch_num):

        mask_list = []

        batch_images = all_images[i*batch_size:(i+1)*batch_size]

        box_pred, mask_pred = model.predict(batch_images)
        # 4: Decode the raw prediction `y_pred`

        masks = mask_pred.reshape([mask_pred.shape[0], mask_pred.shape[1], mask_pred.shape[2]])
        #print(masks.shape)


        y_pred_decoded = decode_detections(box_pred,
                                        confidence_thresh=0.5,
                                        iou_threshold=0.2,
                                        top_k=200,
                                        normalize_coords=normalize_coords,
                                        img_height=img_height,
                                        img_width=img_width)

        #print('shape of y_pred_decoded:', y_pred_decoded)
        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        #print("Predicted boxes:\n")
        #print('   class   conf xmin   ymin   xmax   ymax')
        
        #print(y_pred_decoded[i])

        #########################################################################################################
        images_roi = os.path.join(path, 'dataset', file_name, 'roi', 'roi.jpg')
        #images_roi= 'C:/Users/klickmal/Desktop/8.jpg'
        roi = io.imread(images_roi, as_grey = True)
        # 5: Draw the predicted boxes onto the image
        #print('shape of batch_images:', batch_images[i].shape)
        
        ones = np.ones((masks[0].shape[0], masks[0].shape[1]))
        zeros = np.zeros((masks[0].shape[0], masks[0].shape[1]))

        for j in range(len(y_pred_decoded)):

            masks[j] = np.where(roi, masks[j], roi)
            masks[j] = np.where(masks[j] > 0.1*ones, ones, zeros)
            
            #plt.figure(figsize=(20,12))
            #plt.rcParams['image.cmap'] = 'gray'
            #plt.imshow(masks[i])


            #current_axis = plt.gca()

            #colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist() # Set the colors for the bounding boxes
            classes = ['background', 'car'] # Just so we can print class names onto the image instead of IDs
            #masks[j] = cv2.cvtColor(masks[j], cv2.COLOR_GRAY2BGR) 
            # Draw the predicted boxes in blue
            for box in y_pred_decoded[j]:
                xmin = box[-4]
                ymin = box[-3]
                xmax = box[-2]
                ymax = box[-1]
                #print(xmin)
                #color = colors[int(box[0])]
                label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
                
                #cv2.rectangle(masks[j],(int(xmin),int(ymin)),(int(xmax),int(ymax)),color,1)
                #cv2.putText(masks[j], label, (int(xmin),int(ymin-6)), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5, color)
                cv2.rectangle(batch_images[j],(int(xmin),int(ymin)),(int(xmax),int(ymax)), (55,255,155), 1)
                cv2.putText(batch_images[j], label, (int(xmin),int(ymin-6)), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5, (55,255,155), 1)
            
            mask_list.append(masks[j])
        
        
        mask_list = np.asarray(mask_list)
        #print(mask_list.shape)
        #print(all_images.shape)
        render_images = []
        for k in range(mask_list.shape[0]):
            render_image = draw_mask_on_image_array(batch_images[k], mask_list[k])
            render_images.append(render_image)
            #render_images = np.asarray(render_images)
        '''
        mask_array = np.array([0, 191, 255])
        mask_array = mask_array[np.newaxis, np.newaxis, np.newaxis, :]
        mask_array_tile = np.tile(mask_array, (batch_images.shape[0], batch_images.shape[1], batch_images.shape[2], 1))

        mask_list_new_axis = mask_list[:,:,:, np.newaxis]
        mask_list_tile = np.tile(mask_list_new_axis, (1, 1, 1, 3))
        print(mask_list_tile.shape)
        print(mask_list_tile[0][0])
        render_image = np.where(mask_list_tile, mask_array_tile + batch_images, batch_images)
        '''
        for i in range(mask_list.shape[0]):
            scipy.misc.imsave(mask_saved_dir + '/' + str(id) + '.jpg', mask_list[i])
            scipy.misc.imsave(render_image_dir + '/' + str(id) + '.jpg', render_images[i])
        
        id += 1

        del render_images
        del mask_list
        del render_image

    stop_time = time.time()
    print('time:', stop_time - start_time)

    frames_to_video(render_image_dir, demo_video_dir)       

    

if __name__ == '__main__':

    #dic_dataset = {0: 'winterstreet', 1: 'highway'}
    dic_dataset = {0: 'cnitech_day_time'}
    for i in range(len(dic_dataset)):
        evaluation(dic_dataset[i])