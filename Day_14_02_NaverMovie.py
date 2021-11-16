#Day_14_02_NaverMovie.py
#네이버 영화 리뷰 데이터셋
#김윤박사 깃헙 cnn 이용해서 자연어 처리!
    #깃헙에서 data 들어가서 clean_str 찾아서 써보기!

#위 네이버 코드와 같이 먼저 내 모델을 만들어 놓고 불러와서 사용할 수 있어야 한다!!!



import re
import pandas
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn import preprocessing #이건 1차원데이터 -> 2차원 원핫인코딩으로 바꿔주는것이다.
#퀴즈1: 네이버 영화리뷰 파일로부터 x, y를 반환하는 함수를 만드세요.
def clean_str(string):

    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)

    # string = re.sub(r"[가-힣]", string) #한글 정규표현식
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def get_xy(file_path):
    f = open(file_path, "r", encoding= "utf-8")

    f.readline() #첫줄 읽고 버리기!

    x = []; y = []
    for line in f:
        _, doc, label = line.strip().split('\t')

        x.append(clean_str(doc).split())
        y.append(int(label)) #문자열 -> 숫자형태로!
    f.close()
    small = int(len(x)*0.1)
    return x[:small], np.int32(y[:small])


def save_model():
    x_train, y_train = get_xy("data/ratings_train.txt")
    x_test, y_test = get_xy("data/ratings_test.txt")

    vocab_size = 2000
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x_test)
    tokenizer.fit_on_texts(x_train)
    #퀴즈2
    #X_train에 포함된 토큰들의 길이를 그래프로 표시하세요.
    # length =[]
    # idx = []
    # len([1, 2, 3])
    # x = [i for i in range(len(x_train))]
    # y = sorted([[len(i)]for i in x_train])
    # plt.plot(x, y)
    # plt.show()

    #퀴즈 3
    #앞에서 만든 데이터를 패드까지 추가된 전처리 데이터로 변환하세요!
    max_len = 40
    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_test_seq = tokenizer.texts_to_sequences(x_test) #여기서 숫자로 바뀌게 된 것이다!

    x_train_pad= keras.preprocessing.sequence.pad_sequences(x_train_seq, maxlen=max_len)
    x_test_pad = keras.preprocessing.sequence.pad_sequences(x_test_seq, maxlen=max_len)

    # print(x_train_pad.shape, x_train_pad) #(150000, 40)


    #퀴즈4:  2차원데이터를 원핫이 포함된 3차원 데이터로 변환하세요
    onehots = np.eye(vocab_size, dtype=np.int32) #vocab_size를 feature크기로 가지는 원핫벡터모임 생성

    x_train_onehot = onehots[x_train_pad]
    x_test_onehot = onehots[x_test_pad]
    print(x_train_onehot.shape) #(150000, 40, 2000)
    print(y_train) #(150000, )

    #퀴즈5: test 셋의 정확도를 구하는 모델을 구축하세요

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=x_train_onehot.shape[1:]))  # [max_len, vocab_size]
    model.add(keras.layers.LSTM(50))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.binary_crossentropy,
                  metrics='acc')

    model.fit(x_train_onehot, y_train, epochs=10, batch_size=32, verbose=2,
              validation_data=(x_test_onehot, y_test))

    model.save('data/naver.h5')
    return


def load_model():
    x_train, y_train = get_xy('data/ratings_train.txt')
    x_test, y_test = get_xy('data/ratings_test.txt')
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x_train)

    max_len = 35; vocab_size = 2000
    review = '정말 재미있네.'
    model = keras.models.load_model('model/naver_04_0.45.h5')
    x_review = [clean_str(review).split()]

    x_test_seq = tokenizer.texts_to_sequences(x_review)
    x_test_pad = keras.preprocessing.sequence.pad_sequences(x_test_seq, maxlen=max_len)

    onehots = np.eye(vocab_size, dtype=np.int32)
    x_review_onehot = onehots[x_test_pad]
    print(model.predict(x_review_onehot))
# save_model()
load_model()

#퀴즈: 자신이 작성한 리뷰에 대해 긍정/ 부정 결과를 알려주세요

