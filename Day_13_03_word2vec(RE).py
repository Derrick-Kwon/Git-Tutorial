# Day_13_03_word2vec.py
#클래스 차용하는거 여기서 가져다 써보기!
#RNN의 PreTrained model 이다! 파일이 너무 크기 때문에 다음과 같이 유사한 모델을 만들어해보고
#그다음에 실제 데이터를 가져와서 적용하도록 한다!
#유사모델 : https://word2vec.kr/search/?query=%ED%95%9C%EA%B5%AD-%EC%84%9C%EC%9A%B8%2B%EB%8F%84%EC%BF%84

import numpy as np
import tensorflow.keras as keras
from sklearn import preprocessing
import matplotlib.pyplot as plt


# 관련 사이트
# http://www.rex-ai.info/docs/AI_study_NLP3

# 텍스트 생성 -> 토큰으로 분할(2차원) -> 불용어 제거 -> 단어장 생성 -> 벡터로 변환
# -> skip-gram/cbow 데이터 생성 -> 딥러닝 데이터(원핫)로 변환 -> 모델 구축
def make_vocab_and_vector():
    corpus = ['king is a strong man',
              'queen is a wise woman',
              'boy is a young man',
              'girl is a young woman',
              'prince is a young king',
              'princess is a young queen',
              'man is strong',
              'woman is pretty',
              'prince is a boy will be king',
              'princess is a girl will be queen']

    # 퀴즈
    # 문자열 코퍼스를 토큰으로 구분되어 있는 2차원 코퍼스로 변환하세요
    # 'king is a strong man' -> ['king', 'is', 'a', 'strong', 'man']
    word_tokens = [line.split() for line in corpus]
    # print(word_tokens)    # [['king', 'is', 'a', 'strong', 'man'], ...]

    # 퀴즈
    # 코퍼스로부터 불용어를 제거하세요
    stop_words = ['is', 'a', 'will', 'be']
    word_tokens = [[w for w in tokens if w not in stop_words] for tokens in word_tokens]
    # word_tokens = [w for tokens in word_tokens for w in tokens if w not in stop_words]
    # print(word_tokens)    # [['king', 'strong', 'man'], ...]

    # 퀴즈
    # 단어장을 만드세요
    vocab = {w for tokens in word_tokens for w in tokens}
    vocab = sorted(vocab)
    # print(vocab)
    # ['boy', 'girl', 'king', 'man', 'pretty', 'prince',
    # 'princess', 'queen', 'strong', 'wise', 'woman', 'young']

    # 퀴즈
    # word_tokens를 vocab을 사용해서 숫자로 변환하세요
    # print(word_tokens[0])     # ['king', 'strong', 'man'] -> [2, 8, 3]
    word_vectors = [[vocab.index(w) for w in tokens] for tokens in word_tokens]
    # print(word_vectors)       # [[2, 8, 3], [7, 9, 10], ...]
    print(len(vocab))
    return word_vectors, vocab


def extract(tokens, center, window_size):
    first = max(center - window_size, 0)
    last = min(center + window_size + 1, len(tokens))

    return [tokens[i] for i in range(first, last) if i != center]


def make_xy(word_vectors, vocab, skip_gram):
    xx, yy = [], []
    for tokens in word_vectors:
        # print(tokens)
        for center in range(len(tokens)):
            # print(tokens[center], extract(tokens, center, 1))
            surrounds = extract(tokens, center, 1)
            if skip_gram:
                for target in surrounds:
                    xx.append(tokens[center])
                    yy.append(target)
            else:
                xx.append(surrounds)
                yy.append(tokens[center])

    # print(xx)
    # print(yy)

    # 퀴즈
    # x를 만드세요
    # xx를 활용해서 x를 원핫 벡터로 변환하세요
    x = np.zeros([len(xx), len(vocab)])
    # for i in range(len(xx)):
    #     x[i, xx[i]] = 1

    for i, p in enumerate(xx):
        if skip_gram:
            x[i, p] = 1
        else:
            # print(p)
            onehots = [[int(t == k) for k in range(len(vocab))] for t in p]
            x[i] = np.mean(onehots, axis=0)

    # print(x[:5])

    return x, np.int32(yy)

###########################################################################
#너무 막 나오면 머리가 아프므로 다음과 같이 정리해주도록 한다!(꿀팁!!)
class Epoch100(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 1000 ==0:
            print(epoch, logs['loss'], logs['acc'])

################################################################################

def show_model(x, y, vocab):
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=[len(vocab)]))
    # (28, 2) = (28, 12) @ (12, 2)
    model.add(keras.layers.Dense(2))
    # (28, 12) = (28, 2) @ (2, 12)
    model.add(keras.layers.Dense(len(vocab), activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])
    model.fit(x, y, epochs=4000, verbose=0, callbacks=[Epoch100()])
    model.summary()
    layer1 = model.get_layer(index=0)  #index 0 을 해줬기 때문에 첫번째 dense층의 weight값을 가져온다(중요***!!!)
    w, b = layer1.get_weights()
    print(w, w.shape)

    for token, (x1, x2) in zip(vocab, w):
        print('{:>7} : {} {}'.format(token, x1, x2))
        plt.plot(x1, x2, 'ro')
        plt.text(x1, x2, token) #이걸 사용하면 x1, x2가 사용하는곳에 좌표를 찍어준다.

    w_min = np.min(w, axis=0)
    w_max = np.max(w, axis=0)
    plt.xlim(w_min[0], w_max[0])
    plt.ylim(w_min[1], w_max[1])
    plt.show()       #단어간의 상관관계를 이 그림으로 판단가능하다!!!


word_vectors, vocab = make_vocab_and_vector()
x, y = make_xy(word_vectors, vocab, skip_gram=False)
print(x, y, x.shape, y.shape)
show_model(x, y, vocab)
print(len(vocab))

