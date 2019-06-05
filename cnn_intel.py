#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:23:55 2019

@author: tomallport
"""

import numpy as np                          
import os                                  
from sklearn.metrics import confusion_matrix
import seaborn as sn                       
from sklearn.utils import shuffle          
import matplotlib.pyplot as plt            
import cv2                                  
import tensorflow as tf                     

# Here's our 6 categories that we have to classify.
class_names = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']
class_names_label = {'mountain': 0,
                    'street' : 1,
                    'glacier' : 2,
                    'buildings' : 3,
                    'sea' : 4,
                    'forest' : 5
                    }
nb_classes = 6

def load_data():
    
    datasets = ['seg_train', 'seg_test']
    size = (150,150)
    output = []
    
    for dataset in datasets:
        directory = "./intel-image-classification/" + dataset
        
        images = []
        labels = []
        for folder in os.listdir(directory):
            curr_label = class_names_label[folder]
            for file in os.listdir(directory + "/" + folder):
                img_path = directory + "/" + folder + "/" + file
                curr_img = cv2.imread(img_path)
                curr_img = cv2.resize(curr_img, size)
                images.append(curr_img)
                labels.append(curr_label)
            
                
        images, labels = shuffle(images, labels)     
        images = np.array(images, dtype = 'float32') 
        labels = np.array(labels, dtype = 'int32')   
        
        output.append((images, labels))

    return output


(train_images, train_labels), (test_images, test_labels) = load_data()
train_images = train_images/255.0 
test_images = test_images/255.0 

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)), # the nn will learn the good filter to use
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(input_shape = (150, 150, 3)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(6, activation=tf.nn.softmax)
])
    
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, batch_size=128, epochs=5, validation_split = 0.2)
test_loss = model.evaluate(test_images, test_labels)
