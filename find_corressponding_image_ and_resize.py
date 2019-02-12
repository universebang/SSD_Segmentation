import numpy as np
import tensorflow as tf
import random as rn
import os
from scipy import misc
import scipy
from skimage import io, transform

import keras, glob
from keras.preprocessing import image as kImage

def generateData(train_dir, dataset_dir):
    
    # train_dir: Path to mask groundtruth
    # dataset_dir: path to folder that contains original images

    void_label = -1.
    X_list = []
    Y_list = []
    
    # Given ground-truths, load training frames
    # ground-truths end with '*.png'
    # training frames end with '*.jpg'
    print('bis hier')
    # scan over FgSegNet_dataset for groundtruths
    for root, _, _ in os.walk(train_dir):
        print('train_dir:', train_dir)
        gtlist = glob.glob(os.path.join(root,'*.png'))
        if gtlist:
            Y_list =  gtlist   
            #print(Y_list)
    # scan over CDnet2014_dataset for .jpg files
    for root, _, _ in os.walk(dataset_dir):
        #print('dataset_dir:', dataset_dir)
        #print('root', root)
        inlist = glob.glob(os.path.join(root,'*.jpg'))
        if inlist:
            X_list =  inlist
            #print(X_list)
    
    # filter matched files        
    X_list_temp = []
    for i in range(len(Y_list)):
        Y_name = os.path.basename(Y_list[i])
        Y_name = Y_name.split('.')[0]
        Y_name = Y_name.split('gt')[1]
        for j in range(len(X_list)):
            X_name = os.path.basename(X_list[j])
            X_name = X_name.split('.')[0]
            X_name = X_name.split('in')[1]
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
        x = kImage.load_img(X_list[i])
        x = kImage.img_to_array(x)
        X.append(x)
        
        del x
        x = kImage.load_img(Y_list[i], grayscale = True)
        x = kImage.img_to_array(x)
        shape = x.shape
        x /= 255.0
        x = x.reshape(-1)
        idx = np.where(np.logical_and(x>0.25, x<0.8))[0] # find non-ROI
        if (len(idx)>0):
            x[idx] = void_label
        x = x.reshape(shape)
        x = np.floor(x)
        Y.append(x)
    #print('x:', x)
    #del Y_list, X_list, x, idx
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y

def process_image():
    train_dir = '/home/myubuntu/FgSegNet-master/FgSegNet-master/FgSegNet_dataset2014/nightVideos/winterStreet200'
    dataset_dir = '/home/myubuntu/FgSegNet-master/FgSegNet-master/CDnet2014_dataset/nightVideos/winterStreet/input'
    original, mask = generateData(train_dir, dataset_dir)
    print('shape of mask:', mask.shape)
    print('shape of original:', original.shape)
    print('type of mask:', type(mask))

    num_image, height, width, channel = mask.shape
    mask_new = np.reshape(mask, (num_image, height, width))

    for i in range(mask_new.shape[0]):

        mask_single = transform.resize(mask_new[i], (300, 480))
        original_single = transform.resize(original[i]/255, (300, 480, 3))

        scipy.misc.imsave('/home/myubuntu/Desktop/mask/' + str(i) + '.jpg', mask_single)
        scipy.misc.imsave('/home/myubuntu/Desktop/original/' + str(i) + '.jpg', original_single)

if __name__ == '__main__':
    process_image()

    