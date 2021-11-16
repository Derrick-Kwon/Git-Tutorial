# Day_14_02_NaverMovie.py
import re
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np


# 퀴즈
# 네이버 영화리뷰 파일로부터 x, y를 반환하는 함수를 만드세요
# x는 문자열 토큰, y는 정수로 반환합니다
def get_xy(file_path):
    f = open(file_path, 'r', encoding='utf-8')

    # skip header
    f.readline()

    x, y = [], []
    for line in f:
        # print(line.strip().split('\t'))
        _, doc, label = line.strip().split('\t')

        x.append(clean_str(doc).split())
        y.append(int(label))

    f.close()
    # small = int(len(x) * 0.1)
    # return x[:small], np.int32(y[:small])

    return x, np.int32(y)


def clean_str(string):
    string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def save_model():
    x_train, y_train = get_xy('data/ratings_train.txt')
    x_test, y_test = get_xy('data/ratings_test.txt')

    # 퀴즈
    # x_train에 포함된 토큰들의 길이를 그래프로 표시하세요
    # heights = sorted([len(t) for t in x_train])
    # plt.plot(heights)
    # plt.show()

    vocab_size = 2000
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x_train)

    # 퀴즈
    # 앞에서 만든 데이터를 패드까지 추가된 전처리 데이터로 변환하세요
    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_test_seq = tokenizer.texts_to_sequences(x_test)
    # print(x_train_seq[:3])    # [[25, 897, 7, 1093], [593], []]

    max_len = 40
    x_train_pad = keras.preprocessing.sequence.pad_sequences(x_train_seq, maxlen=max_len)
    x_test_pad = keras.preprocessing.sequence.pad_sequences(x_test_seq, maxlen=max_len)
    # print(x_train_pad.shape, x_test_pad.shape)    # (150000, 35) (50000, 35)
    # print(x_train_pad[:3])

    # onehots = np.eye(10, dtype=np.int32)
    # print(onehots, end='\n\n')
    #
    # print(onehots[[3, 1, 8]], end='\n\n')
    #
    # idx = [[3, 1, 8],
    #        [2, 9, 4]]
    # idx = np.int32(idx)
    # print(onehots[idx], end='\n\n')
    # print(onehots[idx].shape, end='\n\n')

    # 퀴즈
    # 2차원 데이터를 원핫이 포함된 3차원 데이터로 변환하세요
    onehots = np.eye(vocab_size, dtype=np.int32)

    x_train_onehot = onehots[x_train_pad]
    x_test_onehot = onehots[x_test_pad]
    # print(x_train_onehot.shape)       # (150000, 35, 2000)

    # 퀴즈
    # test 셋의 정확도를 구하는 모델을 구축하세요
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=x_train_onehot.shape[1:]))    # [max_len, vocab_size]
    model.add(keras.layers.LSTM(50))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.binary_crossentropy,
                  metrics='acc')

    checkpoint = keras.callbacks.ModelCheckpoint('model/naver_{epoch:02d}-{val_loss:.2f}.h5',
                                                 # 3.중요*** #h5는 케라스에서 저장하는 확장자 #학습이 완료된 모델을 저장하는 것이다.
                                                 save_best_only=True)  # 중요! 용량부담을 줄이기 위하여 다음과 같은 saved_best_eary를 저장한다!!!
    #call back기능 가장 좋은 부분을 가져온다!

    model.fit(x_train_onehot, y_train, epochs=10, batch_size=32, verbose=2,
              validation_data=(x_test_onehot, y_test))


def load_model():
    x_train, y_train = get_xy('data/ratings_train.txt')
    x_test, y_test = get_xy('data/ratings_test.txt')

    vocab_size = 2000
    max_len = 35

    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x_train)
    x_test_seq = tokenizer.texts_to_sequences(x_test)
    x_test_pad = keras.preprocessing.sequence.pad_sequences(x_test_seq, maxlen=max_len)

    onehots = np.eye(vocab_size, dtype=np.int32)
    x_test_onehot = onehots[x_test_pad]


    model = keras.models.load_model('data/naver.h5')
    print(model.evaluate(x_test_onehot, y_test, verbose=0))


save_model()
load_model()
