#콘볼루션 레이어 1~2개 랑 pool 레이어를 잘 짜집기 해야한다.
#hx = wx + b
#2 2   = 2 ,4 4 ,2
#augmentation(증강) 입력사이즈는 224x224 로 하자.

#데이터 100000장을 사용 xx 데이터 1000장과 data augmentation을 이용해보자!
# 1.위치변화  2.색변화

#3x3 5x5등 피쳐의 크기는 '수용력' 이라고 한다. (중요***) - 우리는 무조건 3x3 을 이용하도록 하자!!!!!

import tensorflow.keras as keras
import numpy as np

def mnist_VGG16():
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=[224, 224, 3]))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1, padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1, padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same'))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=1, padding='same',activation='relu'))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=1, padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same'))
    model.add(keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=1, padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=1, padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=1, padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same'))
    model.add(keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1, padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1, padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1, padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same'))
    model.add(keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1, padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1, padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1, padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same'))
    #Dense 레이어는 2차원 데이터가 들어간다. but conv 는 4차원이 들어간다. 따라서 중간에 중간다리가 있어야 한다.
    # model.add(keras.layers.Reshape([-1]))
    model.add(keras.layers.Flatten()) #4차원 -> 2차원
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dense(1000, activation='softmax'))
    model.summary()
    exit()
    return
mnist_VGG16()