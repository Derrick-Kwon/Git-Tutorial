#소설을 쓴다는 것? A라는 단어를 이야기 했을 때, 그 이후에 나올 말이
#말이 되는 것!
#내가 왜 그렇게 생각했을지 생각해보기
#밑의 코드는 character RNN 의 ;연장 코드다
import pandas as pd
import nltk
import tensorflow.keras as keras
from sklearn import preprocessing
import numpy as np

# 퀴즈_10_01_01 : 니체 파일을 x, y 데이터로 변환하는 함수를 만드세요!


def make_data(seq_length):
    f = open('data/nietzsche.txt', 'r', encoding='utf-8')
    nietzsche = f.read() #한줄씩 readline, 한번에 readlines, 기행문자 빼고 read
    nietzsche = nietzsche[:10000]
    nietzsche = nietzsche.lower()
    f.close()


    bin = preprocessing.LabelBinarizer()
    onehot = bin.fit_transform(list(nietzsche))
    # print(onehot.shape) #(10000, 46)
    # print(bin.classes_)

    grams = nltk.ngrams(onehot, seq_length + 1)
    grams = np.float32(list(grams))
    # 'zip' object is not subscriptable 해결할라면 list로 바꾼다
    # print(grams.shape, grams[0]) #(9940, 61, 46)

    #퀴즈 grams를 x, y로 분할하세요
    # x = []
    # y = []
    # for w in grams:
    #     xx = w[:-1]
    #     yy = w[1:]
    #
    #     yy = np.argmax(yy, axis=1)
    #     x.append(xx)
    #     y.append(yy)
    #
    # x = np.float32(x)
    # y = np.float32(y)
    #위에가 내 코드

    x = grams[:, :-1, :]  #3차원이므로 다음과 같이 쓸수 있는데 마지막거는 생략가능하다!
    y = np.argmax(grams[:, -1], axis=1)
    print(x.shape, y.shape)
    return x, y, bin.classes_



#퀴즈_10_01_02: 모델을 구축해서 결과를 예측하세요

def make_model(vocab_size):

    # y = y.reshape(9940, 1) #이건 틀린거다 logistic_regression과 헷갈린거 같은데 그 맞게 바꾸려면(9940, 46) 이렇게 바뀌는게 맞다


    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=x.shape[1:]))
    model.add(keras.layers.GRU(128, return_sequences=False))
    model.add(keras.layers.Dense(vocab_size, activation='softmax')) #(지금은 10000개지만)input_data 크기가 달라질수도 있기 때문에 46을 적으면 안된다!
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    return model

def predict_basic(model, x, vocab):
    print(x.shape)
    p = model.predict(x)
    # print(p, p.shape) #(9940, 46)

    arg_p = np.argmax(p, axis=1)
    print(arg_p)

    print(vocab, ''.join(vocab[arg_p]))
    return

def predict_by_argmax(model, tokens, vocab):
    # print(tokens.shape) #(60, 46) 2차원이다.

    for i in range(100):
        p = model.predict(tokens[np.newaxis])#지금 tokens는 numpy인데 list로 감싸면 에러가 날 수도 있다.
        p = p[0] #token화를 3차원으로 만들었으니 p를 꺼내줌
        arg_p = np.argmax(p)
        print(vocab[arg_p], end='')

    # 60 + 1 #전체 토큰의 길이는 61이 된다. 따라서 이 함수에 계속 넣기 위해선 60으로 맞춰줘야 한다.
    #1+ 59 + 1로 바꾼뒤, 맨 앞에 1을 버린다(가장 영향력이 적은 data를 버려쥼)
        tokens[:-1] = tokens[1:]
        tokens[-1] = p #이 코드는 한칸씩 땡기는 코드이다
    return


seq_length = 60
x, y, vocab = make_data(seq_length)
vocab_size = len(vocab)
model = make_model(vocab_size)
model.fit(x, y, epochs=20, batch_size=64)


print(len(x))
pos = np.random.randint(0, len(x) - seq_length, 1)
pos = pos[0]
tokens = x[pos]

predict_by_argmax(model, tokens, vocab)




#퀴즈 _10_01_03: tokens 부터  시작해서 100 글자를 예측하고 결과를 출력하세요
#에러 빨간색 줄들에서 익숙한 함수를 찾으시오!