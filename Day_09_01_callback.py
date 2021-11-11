#Day_09_01_callback
import random
import tensorflow.keras as keras
from sklearn import preprocessing, model_selection
import numpy as np
import matplotlib.pyplot as plt
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

questions, expected = make_data(size=5000, digits=3)
vocab = '#+0123456789'
chr2idx = {c: i for i, c in enumerate(vocab)}
idx2chr = {i: c for i, c in enumerate(vocab)}

print(questions[:3], expected[:3])
x = make_onehot(questions, chr2idx)
y = make_onehot(expected, chr2idx)

# print(x.shape, y.shape) #(50000, 7, 12)/(50000, 4, 12)
print(chr2idx)
print(idx2chr)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7, shuffle=False)
model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=x.shape[1:]))
model.add(keras.layers.LSTM(128, return_sequences=False))
model.add(keras.layers.RepeatVector(y.shape[1]))
model.add(keras.layers.LSTM(128, return_sequences=True))
model.add(keras.layers.Dense(y.shape[-1], activation='softmax'))
model.compile(optimizer=keras.optimizers.Adam(0.01),
              loss=keras.losses.categorical_crossentropy,
              metrics='acc')
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10) #1. 오버피팅이 나기 시작하면 탈출하는 함수
plateau = keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1) #2. 구불구불한 러닝을 평탄화 작업

checkpoint = keras.callbacks.ModelCheckpoint('model/addition_{epoch:02d}-{val_loss:.2f}.h5',  #3.중요*** #h5는 케라스에서 저장하는 확장자 #학습이 완료된 모델을 저장하는 것이다.
                                             save_best_only=True)  #중요! 용량부담을 줄이기 위하여 다음과 같은 saved_best_eary를 저장한다!!!

history = model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=2, validation_data=(x_test, y_test),
                    callbacks=[checkpoint])



model = keras.models.load_model('model/addition_18-0.99.h5') #이런식으로 학습된 모델을 그냥 불러와서 사용가능하다(단 데이터는 있어야함!)
print(model.evaluate(x_test, y_test))






#퀴즈 09_01_01: loss와 acc그래프를 하나의 피겨에 두개의 플랏으로 그려주세요.

# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], 'r')
# plt.plot(history.history['val_loss'], 'g')
# plt.legend()
# plt.title('loss')
#
#
# plt.subplot(1, 2, 2)
# plt.plot(history.history['acc'], 'r')
# plt.plot(history.history['val_acc'], 'g')
# plt.legend()
# plt.title('acc')
# plt.show()


