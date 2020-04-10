#!/usr/bin/python
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from keras.preprocessing import image
import os
import math

model = models.load_model('Handgesture_model.h5')
gesture = ("down", "palm", "l", "fist", "fist_moved", "thumb", "index", "ok", "palm_moved", "c")

def loop():
    cap = cv2.VideoCapture(0)
    while(True):
        ret,frame =cap.read()
        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # 双边滤波
        frame = cv2.flip(frame, 1)  # 翻转  0:沿X轴翻转(垂直翻转)  大于0:沿Y轴翻转(水平翻转)   小于0:先沿X轴翻转，再沿Y轴翻转，等价于旋转180°
        cv2.rectangle(frame, (int(0.6 * frame.shape[1]), 0),(frame.shape[1], int(0.4 * frame.shape[0])), (0, 0, 255), 2)
        img = frame[0:int(0.4 * frame.shape[0]),int(0.6 * frame.shape[1]):frame.shape[1]]  # 剪切右上角矩形框区域
        
        modes = 'S'
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


        img = cv2.resize(img,(128,128))
        x = np.expand_dims(img, axis = 0) 
        x = x.reshape(1,128,128,1)
        #img_data = np.array(img,dtype = 'float32')
        #img_data = img_data.reshape((1,128,128,1))

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

def analysis(path):
    #img = image.load_img(path, color_mode="grayscale", target_size=(128, 128))
    #x = image.img_to_array(img)  
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128,128))
    cv2.imshow('re',img)
    x = np.expand_dims(img, axis = 0)   
    x = x.reshape(1,128,128,1)
    custom = model.predict(x)
    print(custom[0])

    #draw chart
    y_pos = np.arange(len(gesture))  
    plt.bar(y_pos, custom[0], align='center', alpha=1)
    plt.xticks(y_pos, gesture)
    plt.ylabel('percentage')
    plt.title('gesture')    
    plt.show()

def ana():
    for dirname, _, filenames in os.walk('./test'):
        for filename in filenames:
            path = os.path.join(dirname, filename)
            if path.endswith("png"):
               #img = cv2.imread(path)
               #cv2.imshow('ori',img)
               analysis(path)
if __name__ == '__main__':
   #ana()
   loop()
