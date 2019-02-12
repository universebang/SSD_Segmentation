import cv2
import numpy as np
import os

def frames_to_video(inputpath, outputpath, fps = 13):
   image_array = []
   files = [f for f in os.listdir(inputpath) if os.path.isfile(os.path.join(inputpath, f))]
   files.sort(key = lambda x: int(x[:-4]))#对输入的文件进行排序

   print(files)
   for i in range(len(files)):
       img = cv2.imread(inputpath + '/' + files[i])
       size =  (img.shape[1], img.shape[0])
       img = cv2.resize(img,size)
       image_array.append(img)
   fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
   out = cv2.VideoWriter(outputpath,fourcc, fps, size)
   for i in range(len(image_array)):
       out.write(image_array[i])
   out.release()

def convert():
   inputpath = 'E:/0525/new_Version/dataset/cnitech_day_time/rendering'
   outpath =  'E:/0525/new_Version/dataset/cnitech_day_time/video.avi'
   fps = 13
   frames_to_video(inputpath,outpath,fps)

if __name__ =='__main__':
    convert()