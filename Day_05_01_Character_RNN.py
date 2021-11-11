#Day_05_01_character_RNN
#우리는 텐서플로 ㄴ 케라스로 할 것이다.
#RNN : Recurrent Neural Network : 시계열(순서가 있음)

import tensorflow.keras as keras
from sklearn import preprocessing, model_selection
import numpy as np
#tensor 에서 시작!  -> 안쪽에 x 와 y가 있다!

#tensor #tense => ensor

#퀴즈
#tensor를 원핫 벡터로 변환하고, x, y로 분할하고 해당 데이터로 동작하는 모델을 만드세요


#


def char_rnn_1_dense():

    x = [[1,0,0,0,0,0],    #tense
         [0,1,0,0,0,0],
         [0,0,1,0,0,0],
         [0,0,0,1,0,0],
         [0,0,0,0,1,0]]
    y = [[0,1,0,0,0,0],
         [0,0,1,0,0,0],
         [0,0,0,1,0,0],
         [0,0,0,0,1,0],
         [0,0,0,0,0,1]]

    model = keras.Sequential()
    model.add(keras.layers.Dense(6, activation='softmax'))
    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss = keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    model.fit(x, y)
    return


def char_rnn_1_sparse():
    x = [[1, 0, 0, 0, 0, 0],  # tense
         [0, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0]]
    y =  [1, 2, 3, 4, 5] #ensor

    model = keras.Sequential()
    model.add(keras.layers.Dense(6, activation='softmax'))
    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    model.fit(x, y, epochs=10)
    return
char_rnn_1_sparse()

#but 이 코드로는 앞쪽에서 배운 내용을 뒤로 보낼 수 없다.