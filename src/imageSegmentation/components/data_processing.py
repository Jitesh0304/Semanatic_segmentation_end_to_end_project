import os
import cv2
import numpy as np
from keras.utils import to_categorical
from src.imageSegmentation.entity.config_entity import DataProcessingConfig
from pathlib import Path



class DataPreprocess:
    def __init__(self, config: DataProcessingConfig):
        self.config = config


    def convert_org_img_to_array(self):
        org_folder = self.config.training_org_img_folder
        x = []
        for file in os.listdir(org_folder):
            path1 = os.path.join(org_folder, file)
            img1 = cv2.imread(path1, 1)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img1 = cv2.resize(img1, (512, 512))
            x.append(img1)
        x = np.array(x)
        return x

    def convert_seg_img_to_array(self):
        seg_folder = self.config.training_segment_img_folder
        y = []
        for file2 in os.listdir(seg_folder):
            path2 = os.path.join(seg_folder, file2)
            img2 = np.array(cv2.imread(path2, 1))
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img2 = cv2.resize(img2, (512, 512))
            y.append(img2)
        y = np.array(y)
        return y


    @staticmethod
    def rgb_to_2D_label(image):
        image_seg = np.zeros(image.shape, dtype= np.uint8)

        image_seg[np.all(image == np.array([255, 235, 0]), axis= -1)] = 0
        image_seg[np.all(image == np.array([0, 0 ,0]), axis= -1)] = 1
        image_seg[np.all(image == np.array([27 ,71, 151]), axis= -1)] = 2
        image_seg[np.all(image == np.array([111, 48, 253]), axis= -1)] = 3
        image_seg[np.all(image == np.array([137, 126, 126]), axis= -1)] = 4
        image_seg[np.all(image == np.array([201, 19, 223]), axis= -1)] = 5
        image_seg[np.all(image == np.array([254, 233, 3]), axis= -1)] = 6
        image_seg[np.all(image == np.array([255, 0, 29]), axis= -1)] = 7
        image_seg[np.all(image == np.array([255, 159, 0]), axis= -1)] = 8
        image_seg[np.all(image == np.array([255, 160, 1]), axis= -1)] = 9

        image_seg = image_seg[:,:,0]
        return image_seg
    
    
    def rgb_to_class_num(self):
        segArray = self.convert_seg_img_to_array()
        labels_list = []                             
        for i in range(segArray.shape[0]):
            label = self.rgb_to_2D_label(segArray[i])
            labels_list.append(label)
        labels_arr = np.array(labels_list)
        labels_arr = np.expand_dims(labels_arr, axis=3)
        return labels_arr
    
    def convert_to_categorical(self):
        imgArray = self.rgb_to_class_num()
        classes = self.config.number_of_classes
        label_categorical = to_categorical(imgArray, num_classes = classes)
        return label_categorical