#Day_05_02_RNN_2.py

import tensorflow.keras as keras
import numpy as np
def char_rnn_2_sorted():
    # tensor -> enorst 이처럼 정렬하면 vocab 이 된다.
    #퀴즈 : vocab 을 만드시오!
    word = 'tensor'
    word_sor = sorted(word) #enorst

    x = [[0, 0, 0, 0, 0, 1],  # tenso
         [1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 1, 0, 0, 0]]
    y =  [0, 1, 4, 2, 3] #ensor

    model = keras.Sequential()
    model.add(keras.layers.Dense(6, activation='softmax'))
    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,  #sparse는 y가 x와 1차원 만큼의 차이가 나도 된다.
                  metrics=['accuracy'])
    model.fit(x, y, epochs=10)



def char_rnn_2_simple_rnn(): #


    word = 'tensor'
    word_sor = sorted(word)

    x = [[0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 1, 0, 0, 0]]
    y =  [0, 1, 4, 2, 3]

    x = np.float32([x])
    y = np.float32([y])


    model = keras.Sequential()
    model.add(keras.layers.SimpleRNN(2, return_sequences=True)) #return_sequence 의 기본은 False 이며, 1개를 반환한다
    #RNN쓰는 2가지 방법
    # 1.5개 들어오면 5개 나가는거(n to n)
    # 2.5개 들어오면 마지막으로 나오는 or 중간에의 한가지만 사용하는것( n to 1)
    # hidden state 은 층 중간중간에 지금까지 학습한 내용(context-문맥)이 있다는 것이다.
    # sequence length -> 순서대로 들어오는 것의 셀 갯수
    model.add(keras.layers.Dense(6, activation='softmax'))
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


char_rnn_2_simple_rnn()

#RNN에서는 x는 3차원데이터로 들어오게 된다.(예시: 단어:1차원, 문장:2차원, 글: 3차원) -> y는 2차원이 된다)
#궁금한점? : sparse_categorical_crossentropy가 다른 loss 와는다르게 차원을 벗겨내는 이유는?
#퀴즈2 predict 함수를 이용하여 직접 정확도를 구하세요


char_rnn_2_sorted

