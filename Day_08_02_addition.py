# Day_08_02_addition
# 235+17= 252
# x: 235 +17
# y: 252
import random
import tensorflow.keras as keras
from sklearn import preprocessing, model_selection
import numpy as np
#퀴즈
#자릿수에 맞는 숫자를 만드는 함수를 구현하세요
#digits: 자릿수

def make_number(digits):
    d = random.randrange(digits) +1
    return random.randrange(10 ** d)
for i in range(10):
    print(make_number(3))


def make_data(size, digits):
    questions, expected, seen = [], [], set()
    while len(questions) < size:
        a = make_number(digits)
        b = make_number(digits)

        key = (a, b) if a > b else (b, a) #이 코드 잘 보자!!! 처음보는 코딩방식이다.
        if key in seen:
            continue
        seen.add(key)

        q = '{}+{}'.format(a, b) #덧셈을 string 이라고 생각하자! / q 의 sequence_length는 7자임!
        q += '#'*(digits*2+1-len(q))
        # print(query)

        v = '{}'.format(str(a+b))
        v +='#'*(digits+1-len(v))
        # print(v)
        questions.append(q)
        expected.append(v)
    return questions, expected

def make_onehot(texts, chr2idx):
    batch_size, seq_length, n_features = len(texts), len(texts[0]), len(chr2idx)
    v = np.zeros([batch_size, seq_length, n_features])

    for i, t in enumerate(texts):
        for j, c in enumerate(t):
            k = chr2idx[c]
            v[i, j, k] = 1

    return v

questions, expected = make_data(size=50000, digits=3)
vocab = '#+0123456789'
chr2idx = {c: i for i, c in enumerate(vocab)}
idx2chr = {i: c for i, c in enumerate(vocab)}

print(questions[:3], expected[:3])
x = make_onehot(questions, chr2idx)
y = make_onehot(expected, chr2idx)

print(x.shape, y.shape) #(50000, 7, 12)/(50000, 4, 12)
print(chr2idx)
print(idx2chr)

#퀴즈: 앞에서 만든 x, y 를 y에 대해 80퍼센트로 학습하고, 20퍼센트에 대해 정확도를 구하여라

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7, shuffle=False)
model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=x.shape[1:]))
model.add(keras.layers.SimpleRNN(128, return_sequences=True))
model.add(keras.layers.Reshape([4, -1]))  #7행 12열 -> 결과가 4행 12열이 나와야 하므로 변환해준다!
#model.add(keras.layers.SimpleRNN(128, return_sequences=False))
#model.add(keras.layers.RepeatVector(y.shape[1]) #이렇게 1개가 나오도록 한 다음 그 앞쪽 레이어를 4번 쌓도록 한다.
model.add(keras.layers.Dense(y.shape[-1], activation='softmax'))
model.compile(optimizer=keras.optimizers.Adam(0.01),
              loss=keras.losses.categorical_crossentropy,
              metrics='acc')
model.fit(x_train, y_train, epochs=30, verbose=2, validation_data=(x_test, y_test))