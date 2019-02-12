from xml.dom.minidom import parse
import xml.dom.minidom
import os
import csv

path = "E:/our_dataset/try/label"
files_xml = os.listdir(path)
print(files_xml)
files_xml.sort(key= lambda x:int(x[:-4]))

with open("E:/our_dataset/try/csv/train_label.csv","w", newline = '') as csvfile: 
    writer = csv.writer(csvfile)
    #先写入columns_name
    writer.writerow(["frame","xmin","xmax","ymin","ymax","class_id"])

    for file_name in files_xml:

        # 使用minidom解析器打开 XML 文档
        image_name = file_name.split('.')[0]+'.jpg'
        #print(file_name.split('.')[0]+'.jpg')
        DOMTree = xml.dom.minidom.parse("E:/our_dataset/try/label/" + file_name)
        collection = DOMTree.documentElement

        # 在集合中获取所有实例信息
        object_car = collection.getElementsByTagName("object")

        #print(object_car)
        for car in object_car:
            bndboxes = car.getElementsByTagName("bndbox")
            for bndbox in bndboxes:
                xmin = bndbox.getElementsByTagName("xmin")[0]
                ymin = bndbox.getElementsByTagName("ymin")[0]
                xmax = bndbox.getElementsByTagName("xmax")[0]
                ymax = bndbox.getElementsByTagName("ymax")[0]

                #print(float(xmin.childNodes[0].data))
                #xmin_data = int(0.5 * float(xmin.childNodes[0].data))
                #ymin_data = int(300/540 * float(ymin.childNodes[0].data))
                #xmax_data = int(0.5 * float(xmax.childNodes[0].data))
                #ymax_data = int(300/540 * float(ymax.childNodes[0].data))

                xmin_data = xmin.childNodes[0].data
                ymin_data = ymin.childNodes[0].data
                xmax_data = xmax.childNodes[0].data
                ymax_data = ymax.childNodes[0].data

                #print('xmin is {0}'.format(xmin.childNodes[0].data))
                #print('ymin is {0}'.format(ymin.childNodes[0].data))
                #print('xmax is {0}'.format(xmax.childNodes[0].data))
                #print('ymax is {0}'.format(ymax.childNodes[0].data))
                writer.writerow([image_name, xmin_data, xmax_data, ymin_data, ymax_data,1])



    
