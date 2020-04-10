#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
import os
import cv2
from PIL import Image
from matplotlib import pyplot as plt
# SKLEARN
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# input image dimensions
img_rows, img_cols = 200, 200
# For grayscale use 1 value and for color images use 3 (R,G,B channels)
img_channels = 1
batch_size = 32
nb_classes = 5
nb_epoch = 10  #25

## Path2 is the folder which is fed in to training model
path2 = './gesture-data'
gesture = ["OK", "NOTHING","PEACE", "PUNCH", "STOP"]

# This function can be used for converting colored img to Grayscale img
# while copying images from path1 to path2
def convertToGrayImg(path1, path2):
    for dirname, _, filenames in os.walk(path1):
        for filename in filenames:
            path = os.path.join(dirname, filename)
            print(os.path.join(dirname, filename))
            if path.endswith("png"):
               img = cv2.imread(path)
               #SkindMask模式处理
               hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
               #Apply skin color range
               low_range = np.array([0, 50, 80])
               upper_range = np.array([30, 200, 255])
               skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

               mask = cv2.inRange(hsv, low_range, upper_range)
               mask = cv2.erode(mask, skinkernel, iterations = 1)
               mask = cv2.dilate(mask, skinkernel, iterations = 1)
               #blur
               mask = cv2.GaussianBlur(mask, (15,15), 1)
               #cv2.imshow("Blur", mask)
               #bitwise and mask original frame
               res = cv2.bitwise_and(img, img, mask = mask)
               # color to grayscale
               img = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

               cv2.imwrite(path2+'/'+filename,img)
               

def modlistdir(path, pattern = None):
    listing = os.listdir(path)
    retlist = []
    for name in listing:
        #This check is to ignore any hidden files/folders
        if pattern == None:
            if name.startswith('.'):
                continue
            else:
                retlist.append(name)
        elif name.endswith(pattern):
            retlist.append(name)            
    return retlist

#  init picture
def initializers():
    imlist = modlistdir(path2)   
    image1 = np.array(Image.open(path2 +'/' + imlist[0])) # open one image to get size
    #plt.imshow(im1)    
    m,n = image1.shape[0:2] # get the size of the images
    total_images = len(imlist) # get the 'total' number of images
    
    # create matrix to store all flattened images
    immatrix = np.array([np.array(Image.open(path2+ '/' + images).convert('L')).flatten()
                         for images in sorted(imlist)], dtype = 'f')    
    print(immatrix.shape)   

    ## Label the set of images per respective gesture type.
    label=np.ones((total_images,),dtype = int)
    
    samples_per_class = int(total_images / nb_classes)
    print("samples_per_class - ",samples_per_class)
    s = 0
    r = samples_per_class
    for classIndex in range(nb_classes):
        label[s:r] = classIndex
        s = r
        r = s + samples_per_class

    data,Label = shuffle(immatrix,label, random_state=2)
    train_data = [data,Label]
     
    (X, y) = (train_data[0],train_data[1])
        
    # Split X and y into training and testing sets    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
     
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_channels)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_channels)    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')    
    # normalize
    X_train /= 255
    X_test /= 255  
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, X_test, Y_train, Y_test

# Load CNN model
def CNN(bTraining = True):
    model = Sequential()    
    model.add(Conv2D(32, (3, 3),
                        padding='same',
                        input_shape=(img_rows, img_cols,img_channels)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
    model.summary()
    # Model conig details
    model.get_config()
   
    if not bTraining :
        model.load_weights('./gesture_weight.h5')      
    return model

def trainmodel(model):
    # Split X and y into training and testing sets
    X_train, X_test, Y_train, Y_test = initializers()

    # Now start the training of the loaded model
    hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                 verbose=1, validation_split=0.2)
    visualizeHis(hist)
    ans = input("Do you want to save the trained weights - y/n ?")
    if ans == 'y':
        filename = input("Enter file name - ")
        fname = str(filename) + ".h5"
        model.save_weights(fname,overwrite=True)
    else:
        model.save("newmodel.h5",overwrite=True)

def visualizeHis(hist):
    # visualizing losses and accuracy
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['accuracy']
    val_acc=hist.history['val_accuracy']
    xc=range(nb_epoch)

    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])

    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)

    plt.show()

def analysis(model,path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (img_rows,img_cols))
    x = np.expand_dims(img, axis = 0)   
    x = x.reshape(1,img_rows,img_cols,img_channels)
    prediction = model.predict(x)
    print(prediction[0])
    #draw chart
    y_pos = np.arange(len(gesture))  
    plt.bar(y_pos, prediction[0], align='center', alpha=1)
    plt.xticks(y_pos, gesture)
    plt.ylabel('percentage')
    plt.title('gesture')    
    plt.show()

def capture(model):
    cap = cv2.VideoCapture(0)
    while(True):
        ret,frame =cap.read()
        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # 双边滤波
        frame = cv2.flip(frame, 1)  # 翻转  0:沿X轴翻转(垂直翻转)  大于0:沿Y轴翻转(水平翻转)   小于0:先沿X轴翻转，再沿Y轴翻转，等价于旋转180°
        cv2.rectangle(frame, (int(0.6 * frame.shape[1]), 0),(frame.shape[1], int(0.4 * frame.shape[0])), (0, 0, 255), 2)
        img = frame[0:int(0.4 * frame.shape[0]),int(0.6 * frame.shape[1]):frame.shape[1]]  # 剪切右上角矩形框区域
       
        modes = 'B'
        if modes == 'B':
            #二进制模式处理
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #将的图像转换为灰度图  
            blur = cv2.GaussianBlur(gray, (5, 5), 2)  #加高斯模糊
            th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
            ret, img = cv2.threshold(th3, 60, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)  #二值化处理
        else:
            #SkindMask模式处理
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            #Apply skin color range
            low_range = np.array([0, 50, 80])
            upper_range = np.array([30, 200, 255])
            skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

            mask = cv2.inRange(hsv, low_range, upper_range)
            mask = cv2.erode(mask, skinkernel, iterations = 1)
            mask = cv2.dilate(mask, skinkernel, iterations = 1)
            #blur
            mask = cv2.GaussianBlur(mask, (15,15), 1)
            #cv2.imshow("Blur", mask)
            #bitwise and mask original frame
            res = cv2.bitwise_and(img, img, mask = mask)
            # color to grayscale
            img = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img,(img_rows,img_cols))
        x = np.expand_dims(img, axis = 0) 
        x = x.reshape(1,img_rows,img_cols,img_channels)

        prediction = model.predict(x)

        cv2.rectangle(frame,(10,12),(160,160),(64,64,64),cv2.FILLED)
        cv2.addWeighted(frame.copy(), 0.4, frame, 0.6, 0, frame)
        ges = ""
        for i in range(len(prediction[0])):
           ges = "%s: %s%s" %(gesture[i],round(prediction[0][i]*100, 2),'%')
           cv2.putText(frame, ges,(10,20+15*i),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.imshow('original', frame)
        cv2.imshow('img',img)

        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    mymodel = CNN(False)
    #trainmodel(mymodel)
    #analysis(mymodel,'./imgs/ssstop1.png')
    capture(mymodel)
 
