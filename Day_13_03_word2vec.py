#Day_13_03_word2vec.py
#목적: 단어간에 유사도를 판별할 수 있는가?
import tensorflow.keras as keras
import pandas as np
from sklearn import preprocessing
import nltk
import numpy as np
def make_vocab_and_vector():
    #관련사이트: http://www.rex-ai.info/docs/AI_study_NLP3

    #텍스트 생성 -> 토큰으로 분할, 불용어제거 -> 단어장생성 -> 벡터로 변환 ->
    # skip-gram/cbow (딥러닝데이터-원핫)생성 -> 모델구축 ->학습

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

    corpus_token = []

    stopwords = ['is', 'a', 'will', 'be']

    for i in corpus:
        i = i.split(' ')
        i = list(i)
        for j in stopwords:
            if j in i:
                i.remove(j)
        # print(i)
        corpus_token.append(i)

    # word_tokens = [[w for w in tokens if w not in stop_words] for tokens in word_tokens]
    # word_tokens = [w for tokens in word_tokens for w in tokens if w not in stop_words] #위에것과 이것의 차이를 알 수 있어야 한다!(중요***)
    # print(corpus_token)

    #퀴즈2 1문자열 코퍼스를 토큰으로 구분된 2차원 corpus로 바꾸고, corpus로 부터 불용어를 제거하세요


    #퀴즈3: vocab 을 만드세요
    vocab = []
    for w in corpus_token:
        for i in w:
            if i not in vocab:
                vocab.append(i)
    # print(vocab)

    #퀴즈4: word_tokens를 vocab 을 사용해서 숫자로 변환하세요
    for w in corpus_token:
        for i in w:
            w[w.index(i)] = vocab.index(i)
    word_vectors = corpus_token
    return word_vectors, vocab


def extract(token_count, center, window_size):
    first = max(center - window_size, 0)
    last = min(center + window_size+1, token_count)
    return [tokens[i] for i in range(first, last) if i != center]


word_vectors, vocab = make_vocab_and_vector()
skip_grams = True
xx, yy = [], []
for tokens in word_vectors:
    print(tokens)
    for center in range(len(tokens)):
        #print(tokens[center], extract(tokens, center, 1))
        #skip gram이 뭔지, cbow가 뭔지
        surround = extract(tokens, center, 1)
    if skip_grams:
        for target in surround:
            xx.append(tokens[center])
            yy.append(target)
    else:
        xx.append(surround)
        yy.append(target)

print(xx, yy)
x = np.zeros([len(xx), len(vocab)])
print(x.shape)
#퀴즈: x를 만드세요!
#퀴즈: xx를 활용해서 x 를 원핫 벡터로 변환하세요!

for i in range(len(xx)):
    x[i, 0] = 1
