#Day_word_RNN
#char_rnn_4 모델을 word 버전으로 수정하세요.
import tensorflow.keras as keras
import numpy as np
from sklearn import preprocessing

sentences = ['jeonju is the most beautiful korea',
             'bibimbap is the most famous food',
             'tomorrow i am going to market']

def make_xy(words):
    # long_text =''.join(words)
    long_text = []
    for word in words:
        for w in word.split():
            # print(w)
            long_text.append(w)
    # long_text = [w for word in words for w in word.split()]  위 코드를 이렇게도 바꿀 수 있다.
    print(long_text)


    enc = preprocessing.LabelBinarizer()
    enc.fit(list(long_text))

    x, y = [], []
    for w in words:
        onehot = enc.transform(w.split())
        xx = onehot[:-1]
        yy = onehot[1:]

        yy = np.argmax(yy, axis=1)
        x.append(xx)
        y.append(yy)
    return np.float32([x]), np.float32(y), enc.classes_ #이거 디코더 만드는데 써야한다.


def char_rnn_4(words):
    x, y, vocab = make_xy(words)
    # print(x.shape, y.shape)   # (3, 5, 11) (3, 5)
    # print(vocab)              # ['c' 'e' 'f' 'l' 'n' 'o' 'r' 's' 't' 'w' 'y']
    print(x, y)
    print(x.shape, y.shape)
    x = x.reshape(3, 5, 15)

    model = keras.Sequential()
    model.add(keras.layers.SimpleRNN(32, return_sequences=True))
    model.add(keras.layers.Dense(x.shape[-1], activation='softmax'))

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x, y, epochs=100, verbose=2)
    print(model.evaluate(x, y, verbose=0))

    # 퀴즈
    # predict 함수를 사용해서 직접 정확도를 구하세요
    p = model.predict(x)
    # print(p.shape)        # (3, 5, 11)
    # print(y.shape)        # (3, 5)

    # for i in range(len(x)):
    #     p_arg = np.argmax(p[i], axis=1)
    #     y_arg = y[i]
    #     print(p_arg)
    #     print(y_arg)
    #
    #     print('acc :', np.mean(p_arg == y_arg))

    p_arg = np.argmax(p, axis=2)
    print(p_arg)      # [[1 3 3 5 9] [5 2 2 1 1] [1 4 7 5 6]]
    print('acc :', np.mean(p_arg == y, axis=1))

    # 퀴즈
    # vocab을 사용해서 예측 결과를 디코딩하세요


char_rnn_4(sentences)
