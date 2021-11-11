#Day_05_03_char_rnn_#
#단어를 x, y로 변환하는 함수를 만드세요
import tensorflow.keras as keras
from sklearn import preprocessing, model_selection
import numpy as np
def make_xy(word):
    word = list(word)
    enc = preprocessing.LabelBinarizer()
    bi_word = enc.fit_transform(word)
    x = bi_word[:-1, :]
    y = bi_word[1:, :]
    return x, y

def char_rnn_3(word):
    x, y = make_xy(word)
    y = np.argmax(y, axis=1)
    x = np.float32([x])
    y = np.float32([y])
    # x = x[np.newaxis]; y = y[np.newaxis] #새로운 차원추가 코드 bu

    print(x, y)
    model = keras.Sequential()
    model.add(keras.layers.SimpleRNN(12, return_sequences=True)) #return_sequence 의 기본은 False 이며, 1개를 반환한다
    #RNN쓰는 2가지 방법
    # 1.5개 들어오면 5개 나가는거(n to n)
    # 2.5개 들어오면 마지막으로 나오는 or 중간에의 한가지만 사용하는것( n to 1)
    # hidden state 은 층 중간중간에 지금까지 학습한 내용(context-문맥)이 있다는 것이다.
    # sequence length -> 순서대로 들어오는 것의 셀 갯수
    model.add(keras.layers.Dense(x.shape[-1], activation='softmax')) #퀴즈3: x.shape[-1]을 넣는 이유는  class 이기 때문이다!
    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    model.fit(x, y, epochs=10)
    p = model.predict(x) #p.shape = (1, 5, 6) 1쪽의 데이터는 5행6열/ 5행6열이아니면, 데이터를 추가할 수 없다!
    arg_p = np.argmax(p[0], axis=1)
    y = y[0] #y.reshape 대신 이렇게 해도 된다!
    print(arg_p)
    print(y)
    print("acc: ", np.mean(arg_p == y))
    return


char_rnn_3('rainbow eyes') #처음엔 'tensor'로 해보자!