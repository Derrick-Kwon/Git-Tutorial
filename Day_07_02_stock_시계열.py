#Day_07_02_stock.py
import tensorflow.keras as keras
import numpy as np
from sklearn import preprocessing, model_selection
import pandas as pd
import nltk
import matplotlib.pyplot as plt
#퀴즈_07_02_1
#stock_daily.csv 파일로부터 x, y를 반환하는 함수를 만드세요
#RNN이므로 x 가 3차원이 되어야 한다 (batch_size, seq_length,n_features) = (32-적당히, 7, 5)

def get_xy():
    names = ['Open','High','Low','Volume','Close']
    stock = pd.read_csv('data/stock_daily.csv', names=names, skiprows=2)
    # print(stock, stock.shape) #(732, 5)

    #정규화
    scalar = preprocessing.MinMaxScaler()
    values = scalar.fit_transform(stock.values)  #values를 사용하면 3d -> 2d로 바꿀 수 있다.

    #뒤집기
    values = values[::-1] #뒤집는 코드
    #scale
    seq_length = 7
    grams = nltk.ngrams(values, seq_length+1)
    grams = np.float32(list(grams)) #원하는 크기로 잘라내는 코드!
    # print(grams.shape) #(725, 8, 5)
    print(grams[0])
    x = np.float32([g[:-1] for g in grams])

    y = np.float32([g[-1, -1:] for g in grams])
    # print(x.shape, y.shape) #(725, 7, 5) (725, 1)
    return x, y, scalar.data_min_[-1], scalar.data_max_[-1]

#앞에서 만든 데이터에 대해서 모델을 구축하시오
def stock_model():
    x, y, data_min, data_max= get_xy()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.8, shuffle=False) #시각화할때 shuffle false안하면 엉망이 된다.
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=x.shape[1:]))
    model.add(keras.layers.SimpleRNN(32, return_sequences=False)) # n to 1이므로
    model.add(keras.layers.Dense(1))  #regress 에서는 activation 쓸 필요 없다
    model.summary() #(none,7,32)/(none, 7, 1) -> (none, 32)/(none, 1)

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.mse,
                  metrics= 'mae') #accuracy는 구분에만 쓴다!
    model.fit(x_train, y_train, epochs=100)
    print(model.evaluate(x_test, y_test))
    p = model.predict(x)
    # 퀴즈 2 : 정답과 예측 결과를 시각화 하세요

    #퀴즈3: #예측 데이터를 원래 데이터로 복구하세요!
    p= data_min + (data_max - data_min) *p
    y_test = data_min +(data_max-data_min) * y_test #preprocessing 역으로 하는 것!

    plt.subplot(1, 2, 1)
    plt.plot(y_test, 'r')

    plt.subplot(1, 2, 2)
    plt.plot(y_test, 'r', label = 'target')
    plt.plot(p, 'g' , label = 'prediction')

    plt.show()

    return

stock_model()

#추가 퀴즈: 80퍼센트의 데이터로 학습하고, 20퍼센트의 데이터에 대해 결과를 예측하세요.


