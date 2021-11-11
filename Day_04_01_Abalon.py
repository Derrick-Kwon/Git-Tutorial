#Day_04_01_Abalon
# 퀴즈 아발론 데이터를 읽어서 80퍼센트 학습, 20퍼센트 결과 예측
#분류 - > 정확도
#uci machinelearning repository 에서 원하는 데이터 불러오기
import tensorflow.keras as keras
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection


def Abalon_onehot():
    abalone = pd.read_csv('data/abalone.data')
    x = abalone.values[:, 3:]
    y = abalone.values[:, 0:3]
    print(x, y)
    print(x.shape, y.shape)
    x = preprocessing.scale(x)

    data = model_selection.train_test_split(x, y, train_size=0.8)
    x_train, x_test, y_train, y_test = data

    model = keras.Sequential()
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(3, activation=keras.activations.softmax))  # softmax 는 전체합 1이다. /soft는 많은 클래스중 비중을 구할때 쓴다.
    model.compile(optimizer=keras.optimizers.SGD(0.01),
                  loss=keras.losses.categorical_crossentropy,  # categorical 은 y 값이 원-핫벡터란 뜻
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10)
    print(model.predict(x_test))
    print(model.evaluate(x_test, y_test))
    return

def Abalon_parse():
    # 1단계: 파일읽기
    abalone = pd.read_csv('data/abalone.data_parse', header=None)
    # 2단계 : 데이터 분리
    x = abalone.values[:, 1:]
    x = np.float32(x)
    y = abalone.values[:, 0]   #1차원으로 만들어야 labelbinarizer 에 들어갈 수 있다.
    enc = preprocessing.LabelBinarizer()
    #labelBinarizer - categorical_crossentropy/ LabelEncoder - sparse_categorical_crossentropy 쌍이다!!
    y = enc.fit_transform(y)
    print(y[:5])

    # 3단계 scailing
    x = preprocessing.scale(x)

    # 4단계 : train과 test 나눠주기
    data = model_selection.train_test_split(x, y, train_size=0.8)
    x_train, x_test, y_train, y_test = data

    model = keras.Sequential()
    model.add(keras.layers.Dense(3, activation='softmax'))
    model.compile(optimizer=keras.optimizers.SGD(),
                  loss = keras.losses.categorical_crossentropy,
                  metrics=['Accuracy'])
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    return
Abalon_onehot()
# Abalon_parse()