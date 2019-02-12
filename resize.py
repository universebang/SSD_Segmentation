from skimage import io, transform, filters
import scipy
import os
import numpy as np


'''
path = "C:/Users/klickmal/Desktop/1.jpg"
files_images = os.listdir(path)
#print(files_images)
files_images.sort(key= lambda x:int(x[:-4]))
print(len(files_images))


for image_name in files_images:
    
    #print('image_name:', image_name)
'''
images_dir = 'C:/Users/klickmal/Desktop/3382.jpg' #+ '/' + image_name
print(images_dir)
batch_images = io.imread(images_dir)
print('shape of image:', batch_images[0][0])
batch_images = transform.resize(batch_images, (300, 480))
scipy.misc.imsave('C:/Users/klickmal/Desktop/' + '9.jpg', batch_images)


