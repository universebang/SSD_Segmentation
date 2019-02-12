'''
This module is modified for the input of ssds(ssd + sgmentation) by WK Yang

The base network is from Pierluigi Ferrari

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
import numpy as np
import inspect
from collections import defaultdict
import warnings
import sklearn.utils
from copy import deepcopy
from PIL import Image
import cv2
import csv
import os
import sys
import glob
from tqdm import tqdm, trange
from sklearn.utils import compute_class_weight
from keras.preprocessing import image as kImage

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from data_generator.object_detection_2d_image_boxes_validation_utils import BoxFilter


class Data_Input_Class:
    '''
    A class to get batches of samples, corresponding labels and mask_groundtruth indefinitely.
 
    '''
    def __init__(self,
                 filenames=None,
                 labels_filename = None,
                 mask_groundTruth_Path = None,
                 input_format = ['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                 filenames_type='text',
                 images_dir=None,
                 labels=None,
                 image_ids=None,
                 eval_neutral=None,
                 labels_output_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax'),
                 verbose=True):
        '''
        
        '''
        self.labels_output_format = labels_output_format
        self.labels_format={'class_id': labels_output_format.index('class_id'),
                            'xmin': labels_output_format.index('xmin'),
                            'ymin': labels_output_format.index('ymin'),
                            'xmax': labels_output_format.index('xmax'),
                            'ymax': labels_output_format.index('ymax')} # This dictionary is for internal use.

        self.dataset_size = 0 # As long as we haven't loaded anything yet, the dataset size is zero.
        #self.load_images_into_memory = load_images_into_memory
        self.images = None # The only way that this list will not stay `None` is if `load_images_into_memory == True`.

        #self.class_weights = []  #additional list for Project, not the original parts of keras
        self.cls_weight_list = []  #additional list for Project, not the original parts of keras
        self.images_dir = images_dir
        self.labels_filename = labels_filename
        self.input_format = input_format
        self.mask_groundTruth_Path = mask_groundTruth_Path
        self.images = []
        self.filenames = []

        self.dataset_size = 0
        self.dataset_size = len(self.filenames)
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)

    def parse_csv(self,
                  include_classes='all',
                  random_sample=False,
                  ret=False,
                  verbose=True):
        '''
        Arguments:
            include_classes (list, optional): Either 'all' or a list of integers containing the class IDs that
                are to be included in the dataset. If 'all', all ground truth boxes will be included in the dataset.
            random_sample (float, optional): Either `False` or a float in `[0,1]`. If this is `False`, the
                full dataset will be used by the generator. If this is a float in `[0,1]`, a randomly sampled
                fraction of the dataset will be used, where `random_sample` is the fraction of the dataset
                to be used. For example, if `random_sample = 0.2`, 20 precent of the dataset will be randomly selected,
                the rest will be ommitted. The fraction refers to the number of images, not to the number
                of boxes, i.e. each image that will be added to the dataset will always be added with all
                of its boxes.
            ret (bool, optional): Whether or not to return the outputs of the parser.
            verbose (bool, optional): If `True`, prints out the progress for operations that may take a bit longer.

        Returns:
            None by default, optionally lists for whichever are available of images, image filenames, labels, and image IDs.
        '''
        self.include_classes = include_classes

        # Before we begin, make sure that we have a labels_filename and an input_format
        if self.labels_filename is None or self.input_format is None:
            raise ValueError("`labels_filename` and/or `input_format` have not been set yet. You need to pass them as arguments.")

        # Erase data that might have been parsed before
        
        self.image_ids = []
        self.labels = []

        # First, just read in the CSV file lines and sort them.

        data = []

        with open(self.labels_filename, newline='') as csvfile:
            csvread = csv.reader(csvfile, delimiter=',')
            next(csvread) # Skip the header row.
            for row in csvread: # For every line (i.e for every bounding box) in the CSV file...
                if self.include_classes == 'all' or int(row[self.input_format.index('class_id')].strip()) in self.include_classes: # If the class_id is among the classes that are to be included in the dataset...
                    box = [] # Store the box class and coordinates here
                    box.append(row[self.input_format.index('image_name')].strip()) # Select the image name column in the input format and append its content to `box`
                    for element in self.labels_output_format: # For each element in the output format (where the elements are the class ID and the four box coordinates)...
                        box.append(int(row[self.input_format.index(element)].strip())) # ...select the respective column in the input format and append it to `box`.
                    data.append(box)
        #print('type of data:', type(data))
        #print('length of data:', len(data))
        #
        # print('data[0]:', data[0])
        data = sorted(data) # The data needs to be sorted, otherwise the next step won't give the correct result

        # Now that we've made sure that the data is sorted by file names,
        # we can compile the actual samples and labels lists

        current_file = data[0][0] # The current image for which we're collecting the ground truth boxes
        current_image_id = data[0][0].split('.')[0] # The image ID will be the portion of the image name before the first dot.
        current_labels = [] # The list where we collect all ground truth boxes for a given image
        add_to_dataset = False
        for i, box in enumerate(data):
            #print('data[0]:', data[0])
            if box[0] == current_file: # If this box (i.e. this line of the CSV file) belongs to the current image file
                current_labels.append(box[1:])

                if i == len(data)-1: # If this is the last line of the CSV file
                    if random_sample: # In case we're not using the full dataset, but a random sample of it.
                        p = np.random.uniform(0,1)
                        if p >= (1-random_sample):
                            self.labels.append(np.stack(current_labels, axis=0))
                            self.filenames.append(os.path.join(self.images_dir, current_file))
                            self.image_ids.append(current_image_id)
                    else:
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                        self.image_ids.append(current_image_id)
            else: # If this box belongs to a new image file
                if random_sample: # In case we're not using the full dataset, but a random sample of it.
                    p = np.random.uniform(0,1)
                    if p >= (1-random_sample):
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                        self.image_ids.append(current_image_id)
                else:
                    self.labels.append(np.stack(current_labels, axis=0))
                    self.filenames.append(os.path.join(self.images_dir, current_file))
                    self.image_ids.append(current_image_id)
                current_labels = [] # Reset the labels list because this is a new file.
                current_file = box[0]
                current_image_id = box[0].split('.')[0]
                current_labels.append(box[1:])
                if i == len(data)-1: # If this is the last line of the CSV file
                    if random_sample: # In case we're not using the full dataset, but a random sample of it.
                        p = np.random.uniform(0,1)
                        if p >= (1-random_sample):
                            self.labels.append(np.stack(current_labels, axis=0))
                            self.filenames.append(os.path.join(self.images_dir, current_file))
                            self.image_ids.append(current_image_id)
                    else:
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                        self.image_ids.append(current_image_id)

        self.dataset_size = len(self.filenames)
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
        if True:           
            if verbose: it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
            else: it = self.filenames
            for filename in it:
                with Image.open(filename) as image:
                    self.images.append(np.array(image, dtype=np.uint8))
                    #print('length of self.images:', len(self.images))

        if ret: # In case we want to return these
            return np.asarray(self.images), np.asarray(self.filenames), np.asarray(self.labels), np.asarray(self.image_ids)
        
        
                

    def get_encoded_boxlabel(self,
                 batch_size=32,
                 label_encoder=None,
                 keep_images_without_gt=False,
                 degenerate_box_handling='remove',
                 mask_groundth_dir = None):
        '''
        Generates batches of samples and (optionally) corresponding labels indefinitely.

        Arguments:
            batch_size (int, optional): The size of the batches to be generated.
            shuffle (bool, optional): Whether or not to shuffle the dataset before each pass.
                This option should always be `True` during training, but it can be useful to turn shuffling off
                for debugging or if you're using the generator for prediction.

            label_encoder (callable, optional): Only relevant if labels are given. A callable that takes as input the
                labels of a batch (as a list of Numpy arrays) and returns some structure that represents those labels.
                The general use case for this is to convert labels from their input format to a format that a given object
                detection model needs as its training targets.
        '''

        self.dataset_size = batch_size

        #############################################################################################
        # Do a few preparatory things like maybe shuffling the dataset initially.
        #############################################################################################
        if degenerate_box_handling == 'remove':
            box_filter = BoxFilter(check_overlap=False,
                                   check_min_area=False,
                                   check_degenerate=True,
                                   labels_format=self.labels_format)

        # Override the labels formats of all the transformations to make sure they are set correctly.
        

        #############################################################################################
        # Generate mini batches.
        #############################################################################################

        current = 0

        batch_X, batch_y = [], []

        if current >= self.dataset_size:
            current = 0

        #########################################################################################
        # Get the images, (maybe) image IDs, (maybe) labels, etc. for this batch.
        #########################################################################################

        # We prioritize our options in the following order:
        # 1) If we have the images already loaded in memory, get them from there.
        # 2) Else, if we have an HDF5 dataset, get the images from there.
        # 3) Else, if we have neither of the above, we'll have to load the individual image
        #    files from disk.
        batch_indices = self.dataset_indices[current:current+batch_size]
        if not (self.images is None):
            for i in batch_indices:
                batch_X.append(self.images[i])
            if not (self.filenames is None):
                batch_filenames = self.filenames[current:current+batch_size]
            else:
                batch_filenames = None
        else:
            batch_filenames = self.filenames[current:current+batch_size]
            for filename in batch_filenames:
                with Image.open(filename) as image:
                    batch_X.append(np.array(image, dtype=np.uint8))

        # Get the labels for this batch (if there are any).
        if not (self.labels is None):
            batch_y = deepcopy(self.labels[current:current+batch_size])
        else:
            batch_y = None

        #print('len(batch_y):', len(batch_y))
        # Get the image IDs for this batch (if there are any).
        if not (self.image_ids is None):
            batch_image_ids = self.image_ids[current:current+batch_size]
        else:
            batch_image_ids = None

        current += batch_size

        #########################################################################################
        # Maybe perform image transformations.
        #########################################################################################

            #########################################################################################
            # Check for degenerate boxes in this batch item.
            #########################################################################################
        for i in range(len(batch_X)):

            if not (self.labels is None):

                xmin = self.labels_format['xmin']
                ymin = self.labels_format['ymin']
                xmax = self.labels_format['xmax']
                ymax = self.labels_format['ymax']

                if np.any(batch_y[i][:,xmax] - batch_y[i][:,xmin] <= 0) or np.any(batch_y[i][:,ymax] - batch_y[i][:,ymin] <= 0):
                    if degenerate_box_handling == 'warn':
                        warnings.warn("Detected degenerate ground truth bounding boxes for batch item {} with bounding boxes {}, ".format(i, batch_y[i]) +
                                        "i.e. bounding boxes where xmax <= xmin and/or ymax <= ymin. " +
                                        "This could mean that your dataset contains degenerate ground truth boxes, or that any image transformations you may apply might " +
                                        "result in degenerate ground truth boxes, or that you are parsing the ground truth in the wrong coordinate format." +
                                        "Degenerate ground truth bounding boxes may lead to NaN errors during the training.")
                    elif degenerate_box_handling == 'remove':
                        batch_y[i] = box_filter(batch_y[i])

        #print('length(batch_y):', len(batch_y))
        #########################################################################################

        # CAUTION: Converting `batch_X` into an array will result in an empty batch if the images have varying sizes
        #          or varying numbers of channels. At this point, all images must have the same size and the same
        #          number of channels.

        batch_X = np.array(batch_X)
        #########################################################################################
        # If we have a label encoder, encode our labels.
        #########################################################################################

        #print('len(self.labels):', len(self.labels))
        #print(self.labels is None)
        #print(label_encoder is None)

        if not (label_encoder is None or self.labels is None):           
            batch_y_encoded = label_encoder(batch_y, diagnostics=False)
            batch_matched_anchors = None

            #print('here')
        
        else:
            batch_y_encoded = None
            batch_matched_anchors = None

            #print('here2')

        #print('length(batch_y):',len(batch_y_encoded))

        return np.asarray(batch_X), np.asarray(batch_y_encoded)
    #########################################################################################
    # additional function, not the original part of kears
    #########################################################################################

    def get_mask_label(self):

        if self.cls_weight_list:
            self.cls_weight_list.clear()
            #dataset_dir = mask_groundth_dir
        
        dataset_dir = self.mask_groundTruth_Path

        void_label = -1.
        X_list = []
        Y_list = self.filenames


        for root, _, _ in os.walk(dataset_dir):
            inlist = glob.glob(os.path.join(root,'*.jpg'))
            #print('inlist:', inlist)
            if inlist:
                X_list =  inlist         
                # filter matched files        
        X_list_temp = []
            
        for i in range(len(Y_list)):
            if i== 0:
                pass
                #print('Y_list[0]:',Y_list[0])
                #print('Y_name:', os.path.basename(Y_list[0]))
            Y_name = os.path.basename(Y_list[i])
            Y_name = Y_name.split('.')[0]
            #Y_name = Y_name.split('gt')[1]
            for j in range(len(X_list)):
                X_name = os.path.basename(X_list[j])
                X_name = X_name.split('.')[0]
                #X_name = X_name.split('in')[1]
                if (Y_name == X_name):
                    X_list_temp.append(X_list[j])
                    #print(X_list_temp)
                    break
        X_list = X_list_temp
        #print(X_list)
        #del X_list_temp, gtlist, inlist
                    
        # process training images
        X = []
        Y = []
        for i in range(0, len(X_list)):
            x = kImage.load_img(X_list[i], grayscale = True)
            x = kImage.img_to_array(x)
            shape = x.shape
            x /= 255.0
            x = x.reshape(-1)
            idx = np.where(np.logical_and(x>0.25, x<0.8))[0] # find non-ROI
            if (len(idx)>0):
                x[idx] = void_label
            x = x.reshape(shape)
            x = np.floor(x)
            X.append(x)
                # get the class weight for balancing the data
        X = np.asarray(X)
        #print('X.shape:', X.shape)
        for i in range(X.shape[0]):
            y = X[i].reshape(-1)

            if i == 0 :
                pass
                #print('shape of y:', y.shape)
                #print( np.where(y!=void_label))

            idx = np.where(y!=void_label)[0]
            if(len(idx)>0):
                y = y[idx]

            lb = np.unique(y) #  0., 1
            cls_weight = compute_class_weight('balanced', lb , y)

            #print('type of cls_weight:', cls_weight)

            class_0 = cls_weight[0]
            class_1 = cls_weight[1] if len(lb)>1 else 1.0
                        
            cls_weight_dict = {0:class_0, 1: class_1}
            #print('cls_weight:', cls_weight_dict)
            self.cls_weight_list.append(cls_weight_dict)
                    
        #print('class_weight:', self.cls_weight_list)

        return np.asarray(X), np.asarray(self.cls_weight_list)
   