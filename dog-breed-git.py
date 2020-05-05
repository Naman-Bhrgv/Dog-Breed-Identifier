# -*- coding: utf-8 -*-
"""
Created on Tue May  5 20:22:16 2020

@author: Nomon
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import keras
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import BatchNormalization, Dense, GlobalMaxPooling2D, Lambda, Dropout, InputLayer, Input
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import PIL
from keras.applications import resnet50
from keras.applications import vgg16
from keras.applications import nasnet

def features(model, data_preprocessor, input_size, data):
    
    input_layer = Input(input_size)
    
    preprocessor = Lambda(data_preprocessor)(input_layer) #used to apply preprocessing function on input images
    
    base_model = model(weights='imagenet', include_top=False,input_shape=input_size)(preprocessor)
    
    max = GlobalMaxPooling2D()(base_model)
    
    feature = Model(inputs = input_layer, outputs = max) 
    
    feature_maps = feature.predict(data, batch_size=128, verbose=1)
    
    return feature_maps

def images_to_array(data_dir, train, img_size):
    """
    Function used to resize images and store them in a numpy array.
    We are also encoding breed of the image using one-hot encoder technique. 
    """
    
    train_size = len(train)
    
    x = np.zeros([train_size, img_size[0], img_size[1], img_size[2]], dtype=np.uint8)
    
    y = np.zeros([train_size,1], dtype=np.uint8)
    
    
    i=0
    
    for j in train:
        
        img_name = j
        
        img_dir = os.path.join(data_dir, img_name)
        
        img_pixels = load_img(img_dir, target_size=img_size)
        
        x[i] = img_pixels
        
        image_breed = sel_img[j]
        
        y[i] = class_to_num[image_breed]
        
        i+=1
    
    #One hot encoder
    
    y = to_categorical(y)
    
    return x, y

#Dataset Directory 
data_dir = 'ENTER DIRECTORY PATH WHERE IMAGES ARE STORED'
 
#Image Labels
label=pd.read_csv('ENTER PATH OF CSV FILE HAVING IMAGE LABELS')

ax=pd.value_counts(label['breed'])

ax.plot(kind='barh',fontsize="50",figsize=(60,120))

train,test = train_test_split(label,test_size=0.2)


dog_breeds = sorted(list(set(label['breed'])))

num_classes = len(dog_breeds)

print(num_classes)

class_to_num = dict(zip(dog_breeds, range(num_classes)))

print(class_to_num)

sel_img = dict() 

for index,row in train.iterrows():
    
    sel_img[row['id']+'.jpg']=row['breed']
    
img_size=(331,331,3)

train_x,ytr=images_to_array(data_dir, sel_img, img_size)


resnet_preprocessor = resnet50.preprocess_input

resnet_features = features(resnet50.ResNet50,resnet_preprocessor,img_size, train_x)

print(resnet_features.shape)

nasnet_preprocessor = nasnet.preprocess_input

nasnet_features = features(nasnet.NASNetLarge,nasnet_preprocessor,img_size, train_x)

vgg_preprocessor = vgg16.preprocess_input

vgg_features = features(vgg16.VGG16,vgg_preprocessor,img_size, train_x)

feature_stack = np.concatenate([resnet_features,
                                 nasnet_features,
                                 vgg_features,], axis=-1)
print('Final feature maps shape: ', feature_stack.shape)

from keras.callbacks import EarlyStopping
callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20,mode='max', restore_best_weights=True)

model_callback = [callback]

input_shape=feature_stack.shape[1:]

model = keras.models.Sequential([
    InputLayer(input_shape),
    Dropout(0.6),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
h = model.fit(feature_stack, ytr,
            batch_size=128,
            epochs=60,
            validation_split=0.2,
            callbacks=model_callback)

sel_img = dict() 

for index,row in test.iterrows():
    
    sel_img[row['id']+'.jpg']=row['breed']
    
test_x,ytest=images_to_array(data_dir, sel_img, img_size)

resnet_features = features(resnet50.ResNet50,resnet_preprocessor,img_size, test_x)

nasnet_features = features(nasnet.NASNetLarge, nasnet_preprocessor, img_size, test_x)

vgg_features = features(vgg16.VGG16, vgg_preprocessor, img_size, test_x)

test_features = np.concatenate([resnet_features,nasnet_features,vgg_features],axis=-1)

print('Test feature maps shape', test_features.shape)

y_pred = model.predict(test_features, batch_size=128)
print(y_pred.shape)

y_pred_oh=list()
c=0

for i in y_pred:
    
    mv=np.amax(i)
    
    a=[]
    
    for j in i:
        
        if j==mv:
            
           a.append(1)
        
        else:
            
            a.append(0)
            
    y_pred_oh.append(a)
    
    c=c+1
ytest=test['breed'].to_list()
l=list()

for i in y_pred_oh:
    
    c=-1
    
    for j in i:
        
        c=c+1
        
        if j==1:
            
            break
    
    
    l.append(dog_breeds[c])
correct=0

for i in range(len(l)):
    
    if l[i]==ytest[i]:
        
        correct+=1
        

acc=correct/len(l)

acc*=100

print("Accuracy of model is ",acc)