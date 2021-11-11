import tensorflow.keras as keras
from sklearn import preprocessing, model_selection
import numpy as np

def make_xy(words):
    long_text = ''.join(words)
    print(long_text)
    enc = preprocessing.LabelBinarizer()
    enc.fit(list(long_text))
    x, y = [], []
    for w in words:
        onehot = enc.transform(list(w)) #fit공부/transform변환인데 fit_transform과 다르게 따로따로한다

        xx = onehot[:-1]
        yy = onehot[1:]

        yy = np.argmax(yy, axis=1)
        x.append(xx)
        y.append(yy)
        # x = np.float32([x]); y = np.float32(y) #이렇게 쓰면 위에 설정한 x, y와 바뀐 x, y가 충돌을 일으킨다
    return np.float32([x]), np.float32(y), enc.classes_ #이거 디코더 만드는데 써야한다.

#플라스크란? : 어떤 모델을 만들던, 스마트폰 or 홈페이지 인터페이스를 붙이면 가산점이 있을 것이다!

def char_rnn_4(words):
    x, y, vocab = make_xy(words)
    x = x[0]
    print(vocab)
    # print(x.shape, y.shape) #(3-batchsize, 5-sequencelength, 11-features,class) (3, 5)
    # x = x[np.newaxis]; y = y[np.newaxis] #새로운 차원추가 코드 bu

    model = keras.Sequential()
    model.add(keras.layers.SimpleRNN(32, return_sequences=True)) #return_sequence 의 기본은 False 이며, 1개를 반환한다
    #RNN쓰는 2가지 방법
    # 1.5개 들어오면 5개 나가는거(n to n)
    # 2.5개 들어오면 마지막으로 나오는 or 중간에의 한가지만 사용하는것( n to 1)
    # hidden state 은 층 중간중간에 지금까지 학습한 내용(context-문맥)이 있다는 것이다.
    # sequence length -> 순서대로 들어오는 것의 셀 갯수
    model.add(keras.layers.Dense(x.shape[-1], activation='softmax')) #퀴즈3: x.shape[-1]을 넣는 이유는  class 이기 때문이다! / softmax 11개중 확률이 가장 높은것을 선택하는 것이다.,
    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    model.fit(x, y, epochs=10)
    p = model.predict(x) #p.shape = (1, 5, 6) 1쪽의 데이터는 5행6열/ 5행6열이아니면, 데이터를 추가할 수 없다!

    for i in range(len(x)):
        p_arg = np.argmax(p[i], axis=1)
        y_arg = y[i]
        print('acc: ', np.mean(p_arg ==y_arg))
    p_arg = np.argmax(p, axis=2)
    print(p_arg)
    print('acc: ', np.mean(p_arg==y), axis = 1)

    #이 p_arg 을 디코딩해서 다시 바꿀 수 있어야 한다.
    # 퀴즈6-1: vocab 을 이용해서 예측 결과를 디코딩하세요.
    for j in p_arg[0]:
        print(vocab[j])
    #comprehension 하는법! 반복문 -> 한줄로   : print([vocab[j] for j in p_arg[0]])

    # for i in range(len(p)):
    #     print([vocab[j] for j in p_arg[i]])

    #위 코드 대신에 이렇게도 가능
    # for pread in p_arg:
    #     print('p:', ''.join([vocab[j]for j in pread]))

    for pred, yy in zip(p_arg, y): #p_arg, y 에서 각각하나씩 pred와 yy에 넣는다.
        print('p:', ''.join([vocab[j] for j in np.int32(yy)])) #인덱스로 쓸 때는 정수로 써야한다.
        print('p:', ''.join([vocab[j] for j in pred])) #인덱스로 쓸 때는 정수로 써야한다.

    print(vocab[p_arg])
    print([''.join(i) for i in vocab[p_arg]]) #위 for문코드를 이렇게 쓸 수도 있다.
    return


char_rnn_4(['yellow', 'coffee', 'tensor'])
#퀴즈: 여러개의 단어를 x, y로 변환하는 함수를 만드세요!



