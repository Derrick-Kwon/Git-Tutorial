import tensorflow.keras as keras
from sklearn import preprocessing, model_selection
import numpy as np

def make_xy(words):
    long_text = ''.join(words)
    print(long_text)
    enc = preprocessing.LabelBinarizer()
    enc.fit(list(long_text))
    x, y = [], []
    pad_num = int(max(len(word) for word in words))
    # print(pad_num) #10
    for w in words:
        if len(w) < pad_num:
            w +='*'*(pad_num-len(w))
        onehot = enc.transform(list(w)) #fit공부/transform변환인데 fit_transform과 다르게 따로따로한다
        xx = onehot[:-1]
        yy = onehot[1:]

        yy = np.argmax(yy, axis=1)
        x.append(xx)
        y.append(yy)
        # x = np.float32([x]); y = np.float32(y) #이렇게 쓰면 위에 설정한 x, y와 바뀐 x, y가 충돌을 일으킨다
    return np.float32([x]), np.float32(y), enc.classes_ #이거 디코더 만드는데 써야한다.


def char_rnn_5(words):
    x, y, vocab = make_xy(words)
    print(vocab)
    x = x.reshape(3, 9, 13)
    # print(x.shape, y.shape) #(3-batchsize, 5-sequencelength, 11-features,class) (3, 5)
    # x = x[np.newaxis]; y = y[np.newaxis] #새로운 차원추가 코드 bu

    model = keras.Sequential()
    model.add(keras.layers.SimpleRNN(32, return_sequences=True)) #return_sequence 의 기본은 False 이며, 1개를 반환한다
    model.add(keras.layers.Dense(x.shape[-1], activation='softmax')) #퀴즈3: x.shape[-1]을 넣는 이유는  class 이기 때문이다! / softmax 11개중 확률이 가장 높은것을 선택하는 것이다.,
    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    model.fit(x, y, epochs=10)
    p = model.predict(x) #p.shape = (1, 5, 6) 1쪽의 데이터는 5행6열/ 5행6열이아니면, 데이터를 추가할 수 없다!
    p_arg = np.argmax(p, axis=2)
    print(vocab[p_arg])
    print([''.join(i) for i in vocab[p_arg]]) #위 for문코드를 이렇게 쓸 수도 있다.
    for i, w in zip(vocab[p_arg], words):
        print(w, len(w))
        valid = len(w) -1
        print(''.join(i[:valid]))

    print('-'*70)
    #여기부터 05_06의 문장을 위한 코드다

    return


if __name__ =='__main__':   #RNN_5를 직접구동할때만 사용하게 해주는 장치!
                            #다른곳에서 임포트 할 때, 임포트 당하는 함수를 이런식으로 처리 해준다!
    char_rnn_5(['yellow','sky','blood_game'])
#[3, ?, 11] -> 단어마다 배열의 숫자가 다르므로  -> [3, 9, 11]
#padding
#잘라서 버릴때는 버려도 괜찮다는 확신이 있어야 한다.
#퀴즈5-1: 길이가 다른 단어들의 목록에 동작하도록 수정하세요(패딩: *)
#퀴즈5-2: 패딩에 대해 예측한 결과를 버리고 출력하시오