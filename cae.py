import torch

import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np

import warnings
import configparser
import os
import csv
import pandas as pd
import numpy as np
import time
from numpy import zeros, newaxis
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D,Dense,Flatten,Reshape ,Dropout
from keras.models import Model, load_model
from keras import regularizers
from keras import backend as K

import matplotlib.pyplot as plt

import os
from PIL import Image

import tensorflow as tf

import os
import numpy as np
import cv2, time, math
from matplotlib import pyplot as plt
from scipy.signal import convolve2d as conv2
#兩個自定義模組（庫）匯入
###from bilateralfilt import bilatfilt
###from dog import deroGauss

## run on GPU
# if torch.cuda.is_available():
#     device_count = torch.cuda.device_count()
    
#     for i in range(device_count):
#         print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
# else:
#     print("not found")

def cae(train,val):
    activation = 'relu'
    
    input_sig = Input(batch_shape=(None, 16,16,3))
    x = Convolution2D(250, (3,3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding='same',
                      kernel_regularizer=regularizers.l2(0.001))(input_sig)
   
    x1 = MaxPooling2D(2,2)(x)
    x2 = Convolution2D(25, (3,3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding='same')(x1)
    encoded = MaxPooling2D(2,2)(x2)
    
    encoded = Dropout(0.25)(encoded)
    encoded = Flatten()(encoded)
    units = encoded.shape[1]

    #encoded = Dropout(0.5)(encoded)

    encoded = Dense(16,activation=tf.keras.layers.LeakyReLU(alpha=0.1),name="latent")(encoded)

    Encodeder = Model(input_sig, encoded)
    #print("shape of encoded {}".format(K.int_shape(encoded)))

    encoded = Dense(units,activation=tf.keras.layers.LeakyReLU(alpha=0.1))(encoded)
    #encoded = Dense(128)(encoded)

    encoded = Reshape((4, 4, 25))(encoded)

    """
    Encodeder = Model(input_sig, encoded)
    print("shape of encoded {}".format(K.int_shape(encoded)))
    """
    #    decode_input = Input(batch_shape=(None,3,3))
    x2_ = Convolution2D(25, (3,3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding='same', kernel_regularizer=regularizers.l2(0.001))(
        encoded)
    x1_ = UpSampling2D(2)(x2_)
    x_ = Convolution2D(250, (3,3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding='same')(x1_)
    upsamp = UpSampling2D(2)(x_)
    # upsamp = Flatten()(upsamp)
    #    upsamp = Dense(40,activation='sigmoid')(upsamp)
    #    Reshape((4,10))(upsamp)
    ##    upsamp = Convolution1D(output_len, window_size, activation='sigmoid', padding='same')(upsamp)
    Encodeder = Model(input_sig, encoded)
    #print("shape of encoded {}".format(K.int_shape(encoded)))
    decoded = Convolution2D(3, (3,3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding='same')(upsamp)
    #    Decodeder = Model(decode_input, decoded)

    #print("shape of decoded {}".format(K.int_shape(decoded)))
    autoencoder = Model(input_sig, decoded)
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['msle'])

    print('autoencoder.summary:\n', autoencoder.summary())
    
    # autoencoder.fit_generator(image_gen.flow(train, train, batch_size=64), epochs=30
                              
    #                                 #, shuffle=False
    #                                 , verbose=1)
    
    
    
    autoencoder.fit(train, train
                    , epochs=50
                    , batch_size=64
                    , shuffle=False
                    , verbose=1  # 日誌顯示，0為不在標準輸出流輸出日誌信息，1為輸出進度條記錄，2為每個epoch輸出一行記錄
                    # , validation_data=(x_test, y_test)
                    )
    
    '''
    Step3.4
    #===計算驗證集的重建誤差===
    '''
    x_val_predict = autoencoder.predict(val)
    #print("val :",val[0][0][0][2])
    #print("x_val_predict :",x_val_predict[0][0][0][2])
    
    def calculate_losses(x, preds):
        losses = np.zeros(len(x))
        for i in range(len(x)):
            losses[i] = ((preds[i] - x[i]) ** 2).mean(axis=None)
        return losses
    
    print('===計算驗證資料集預測後的重建誤差===')
    val_losses = calculate_losses(val, x_val_predict)  # 驗證資料集之重建誤差
    print('val_losses最小值:\n', np.percentile(val_losses, 0))
    print('val_losses最大值:\n', np.percentile(val_losses, 100))
    print('val_losses平均值:\n', np.mean(val_losses))
    
    epochs = range(1, len(val_losses) + 1)
    plt.plot(epochs, val_losses, color='b')
    plt.xlabel('val') # 設定x軸標題
    plt.plot(epochs, val_losses, 'b', label='val_losses')
    plt.title('val_losses') # 設定圖表標題
    plt.show()
    
    #print(val_losses)
    # print('val_losses:\n',val_losses)

    # 將此重建誤差設定為閥值
    # threshold=np.mean(val_losses)  #val_losses的平均值
    # threshold=np.percentile(val_losses,100)  #val_losses的最大值


    threshold = np.percentile(val_losses, 100)  # val_losses的最大值 + 係數調整
    print('重建誤差之閥值=', threshold)
    
    #autoencoder.save("/home/eden/Pictures/0216_canny/CAE_model_964_2.h5")
    
    return autoencoder, threshold

def cae_predict (test,autoencoder,threshold):
    

    x_predict = autoencoder.predict(test)


    '''計算測試資料集預測後的重建誤差'''

    def calculate_losses_b(x, preds):
        losses = np.zeros(len(x))
        for i in range(len(x)):
            losses[i] = ((preds[i] - x[i]) ** 2).mean(axis=None)
        return losses

    test_losses = calculate_losses_b(test, x_predict)  # 驗證資料集之重建誤差

    print("test_losses",test_losses)

    print('test_losses最小值:\n', np.percentile(test_losses, 0))
    print('test_losses最大值:\n', np.percentile(test_losses, 100))
    print('test_losses平均值:\n', np.mean(test_losses))
    
    epochs = range(1, len(test_losses) + 1)
    plt.plot(epochs, test_losses, color='b')
    plt.xlabel('test_') # 設定x軸標題
    plt.plot(epochs, test_losses, 'b', label='test_losses_')
    plt.title('test_losses_') # 設定圖表標題
    plt.show()
    
    
    #print(test_losses)

    new_abnomal_threshold = np.percentile(test_losses, 0) 
    min_losses = np.percentile(test_losses, 5)
    
    '''檢查此重建誤差，如果大於threshold則判斷為異常'''
    y_predict = np.zeros(len(test_losses))  # 對測試集最後的預測結果(兩類)
    y_predict[np.where(test_losses >= threshold)] = 1

    #test_losses = DataFrame(test_losses)
    #y_predict = DataFrame(y_predict)
    
    return y_predict , test_losses


trainPath = "D:/AI/QTR_eden/QTR/dataset/train/"

trainFilePath = os.listdir(trainPath)
train_record = []

for file in trainFilePath:
    train_record.append(file)
#print(train_record)
print(len(train_record))
train_final = []
for i in range(len(train_record)):
    imagepath = "D:/AI/QTR_eden/QTR/dataset/train/"+train_record[i]+""
    #img = cv2.imread("D:/AI/QTR_eden/QTR/dataset/train/"++)

    img = cv2.imread(imagepath)
    img = cv2.resize(img, (16, 16))
    #print(img.shape)
    # cv2.imshow("window_name", img)

    # cv2.waitKey(0)

    # # closing all open windows
    # cv2.destroyAllWindows()
    train_final.append(img)


valPath = "D:/AI/QTR_eden/QTR/dataset/val/"

valFilePath = os.listdir(valPath)
val_record = []

for file in valFilePath:
    val_record.append(file)

print(len(val_record))
val_final = []
for i in range(len(val_record)):
    imagepath = "D:/AI/QTR_eden/QTR/dataset/val/"+val_record[i]+""
    #img = cv2.imread("D:/AI/QTR_eden/QTR/dataset/train/"++)

    img = cv2.imread(imagepath)
    img = cv2.resize(img, (16, 16))
    # cv2.imshow("window", img)

    # cv2.waitKey(0)

    # # closing all open windows
    # cv2.destroyAllWindows()
    val_final.append(img)

#print(len(val_final))
x_train = np.array(train_final).reshape(27,16,16,3)
x_val = np.array(val_final).reshape(3,16,16,3)
#print(len(x_train))
autoencoder,threshold = cae(x_train,x_val)


testPath = "D:/AI/QTR_eden/QTR/dataset/test/"

testFilePath = os.listdir(testPath)
test_record = []

for file in testFilePath:
    test_record.append(file)

print(len(test_record))
test_final = []
for i in range(len(test_record)):
    imagepath = "D:/AI/QTR_eden/QTR/dataset/test/"+test_record[i]+""
    #img = cv2.imread("D:/AI/QTR_eden/QTR/dataset/train/"++)
    
    img = cv2.imread(imagepath)
    img = cv2.resize(img, (16, 16))
    # cv2.imshow("window", img)

    # cv2.waitKey(0)

    # # closing all open windows
    # cv2.destroyAllWindows()
    test_final.append(img)
    
print(test_record)
x_test = np.array(test_final).reshape(2,16,16,3)
#print(x_test)
result_2 , test_losses_1  = cae_predict(x_test,autoencoder,threshold)
print(result_2)
print(test_losses_1)


#####
##Evaluate

#ground truth
label = [1,0]
TP = 0
FP = 0
FN = 0
TN = 0

for z in range(len(result_2)):
    if result_2[z] == 1 and label[z] == 1:
        TP = TP + 1
    elif result_2[z] == 1 and label[z] == 0:
        FP = FP + 1
    elif result_2[z] == 0 and label[z] == 1:
        FN = FN + 1
    else:
        TN = TN + 1


Accuracy = (TP + TN) / (TP+FP+FN+TN)
print('Accuracy:',Accuracy)

Recall = TP / (TP+FN)
print('Recall:',Recall)

Precision = TP / (TP+FP)
print('Precision:',Precision)

F1_score =  2/ ((1/Precision) + (1/Recall))
print('F1-score:',F1_score)  
#####

