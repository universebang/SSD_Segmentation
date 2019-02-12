from skimage import io, transform, filters
import scipy
import os
import numpy as np
import copy



path = "E:/0525/mask_richtig_3"
#path = 'C:/code/python/FgsegNet/FgSegNet_dataset2014/nightVideos/winterStreet200'
files_images = os.listdir(path)
#print(files_images)
files_images.sort(key= lambda x:int(x[:-4]))
print(len(files_images))


for image_name in files_images:
    
    #print('image_name:', image_name)

    images_dir = 'E:/0525/mask_richtig_3' + '/' + image_name
    #images_dir = 'C:/code/python/FgsegNet/FgSegNet_dataset2014/nightVideos/winterStreet200/' + image_name
    
    batch_images = io.imread(images_dir)
    #print('shape of image:', batch_images.shape)
    #batch_images = transform.resize(batch_images, (300, 480, 3))
    #print(batch_images)
    copy_image = copy.deepcopy(batch_images)
    for i in range(300-1):
        for j in range(480-1):
            #print(batch_images.max())
            '''
            if batch_images[i][j] >= 200:
                copy_image[i][j] = 255
            elif batch_images[i][j] <= 20:
                copy_image[i][j] = 128
            else:
                copy_image[i][j] = 0
            '''
            if  copy_image[i][j] >= 200:
                if copy_image[i-1][j-1] <= 150:
                    batch_images[i-1][j-1] = 255
                if copy_image[i-1][j] <= 150:
                    batch_images[i-1][j] = 255
                if copy_image[i-1][j+1] <= 150:
                    batch_images[i-1][j+1] = 255
                if copy_image[i-1][j] <= 150 :
                    batch_images[i-1][j] = 255
                if copy_image[i+1][j] <= 150 :
                    batch_images[i+1][j] = 255
                
                if copy_image[i+1][j-1] <= 150 :
                    batch_images[i+1][j-1] = 255
                if copy_image[i+1][j+1] <= 150 :
                    batch_images[i+1][j+1] = 255
                if copy_image[i+1][j] <= 150 :
                    batch_images[i+1][j] = 255
    scipy.misc.imsave('E:/0525/mask_richtig_4/' + image_name, batch_images)

