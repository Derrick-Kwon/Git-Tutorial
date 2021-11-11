#Day_08_01_google!
# http://finance.yahoo.com/quote/GOOG/history?ltr=1
#1.공모전!
#2.텐서플로 개발자 자격증(google)

#이거 오류 고치기!!!

#퀴즈8-1-1: goog파일을 읽고, 70%로 학습하고 30%에 대해 결과를 예측하세요(close 예측)
import tensorflow.keras as keras
import numpy as np
from sklearn import preprocessing, model_selection
import pandas as pd
import nltk
import matplotlib.pyplot as plt

def get_xy():
    google = pd.read_csv('data/GOOG.csv', index_col=0)
    values = [google['Open'], google['High'], google['Low'], google['Volume'],
             google['Close']]
    values = np.transpose(values)

    scalar = preprocessing.MinMaxScaler()
    values = scalar.fit_transform(values)
    # print(values) #date 제거

    seq_length = 7
    grams = nltk.ngrams(values, seq_length+1)
    grams = np.float32(list(grams))

    print(grams[0])
    return
    x = np.float32([w[:-1] for w in grams])
    y = np.float32([w[-1, -1:] for w in grams])
    return x, y



    grams = nltk.ngrams(values, 7+1)
    grams = np.float32(list(grams))
    # print(grams.shape)            # (725, 8, 5)

    x = np.float32([g[:-1] for g in grams])
    y = np.float32([g[-1, -1:] for g in grams])
    # print(x.shape, y.shape)       # (725, 7, 5) (725, 1)

def stock_model():
    x, y= get_xy()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7, shuffle=False) #시각화할때 shuffle false안하면 엉망이 된다.
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
    p = model.predict(x_test)
    plt.plot(p, 'r')
    plt.plot(y_test, 'g')
    plt.ylim(2000, 3000)

    plt.show()

stock_model()