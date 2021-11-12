from sklearn import datasets, model_selection, preprocessing
import tensorflow.keras as keras
import numpy as np
#피쳐가 몇개 없을때는 오히려 다음 코드들과 같이, Dense를 늘렸다가 다시 줄이는 방향으로 하는 경우가 많다.
#Linnerud 데이터셋에 대해 동작하는 딥러닝 모델을 구축하세요
x, y = datasets.load_linnerud(return_X_y=True)
#print(x.shape, y.shape) #(20, 3), (20, 3)
#multiple regression 모델이다!

x = preprocessing.scale(x)
y = preprocessing.scale(y)

y1 = y[:, :1]
y2 = y[:, 1:2]
y3 = y[:, 2:3]


inputs = keras.layers.Input(shape=[3])
output1 = keras.layers.Dense(6, activation='relu', name='dense1')(inputs)
output1 = keras.layers.Dense(1, name='weights')(output1) #마지막에 softmax나 sigmoid 를 쓰지 않으면 regression 이다!

output2 = keras.layers.Dense(6, activation='relu', name='dense2')(inputs)
output2 = keras.layers.Dense(1, name='waist')(output2)

output3 = keras.layers.Dense(6, activation='relu', name='dense3')(inputs)
output3 = keras.layers.Dense(1, name='pulse')(output3)


model = keras.Model(inputs, [output1, output2, output3])

model.summary()

model.compile(optimizer=keras.optimizers.SGD(0.1),
              loss=keras.losses.mse)
model.fit(x, [y1, y2, y3], epochs=100)


