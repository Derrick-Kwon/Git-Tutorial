#Day_09_02

#퀴즈 car.data 파일을 읽어서 60퍼센트로 학습하고 20퍼센트로 검증하고 최종적으로 나머지 나머지 20퍼센트에 대해 결과를 예측하는 모델을 구축하세요.

import tensorflow.keras as keras
from sklearn import preprocessing, model_selection
import numpy as np
import pandas as pd

names = ['buying', 'maint', 'doors', 'persons', 'lug_boot',
         'safety', 'class']
cars = pd.read_csv('data/car.data',names=names)
print(cars)
enc = preprocessing.LabelBinarizer()  #binarizer 쓰면 안된다! 왜?

buying = enc.fit_transform(cars['buying'].values)
maint = enc.fit_transform(cars['maint'].values)
doors = enc.fit_transform(cars['doors'].values)
persons = enc.fit_transform(cars['persons'].values)
lug_boot = enc.fit_transform(cars['lug_boot'].values)
safety = enc.fit_transform(cars['safety'].values)

x = [buying, maint, doors, persons, lug_boot, safety]
x = np.transpose(x)
x = preprocessing.scale([x])
y = enc.fit_transform(cars['class'].values)
y = preprocessing.scale([y])

print(x.shape, y.shape)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.8)

model = keras.Sequential()
model.add(keras.layers.Dense(4, activation='softmax'))
model.compile(optimizer=keras.optimizers.Adam(0.1),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['Accuracy'])
model.fit(x_train, y_train, epochs=10,
          validation_split=0.75,
          batch_size=32)


model_car_dense
쓰레기

#feature 가 늘어나면 (원핫으로 하면) 더 성능이 높아진다. -> 집에가서 해봐라 ㅡㅡ
