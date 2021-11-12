#중심단어, 주변단어 묶는다는 것은 관련이 있다는 것이다. -> 이 관련이 있다는 것끼리 묶어서 학습하면 ... 뭉탱이로 몰려있게 된다.
#(주변단어 하나하나와 중심단어) - >skip-grams, (주변단어 전체 -> 중심단어예측) -> cbow
#데이터의 갯수가 많기 때문에

#퀴즈 13_02_01 :주변단어 인덱스만 추출하는 함수를 만드세요
import nltk
import numpy as np
def extract(token_count, center, window_size):
    first = max(center - window_size, 0)
    last = min(center + window_size+1, token_count)
    return [i for i in range(first, last) if i != center]

def show_word2vec(tokens,skipgram):
    for center in range(len(tokens)):
        surrounds = extract(len(tokens), center, 2)

        # print(extract(len(tokens), center, 2), center)

        # 퀴즈: skip-gram 방식으로 결과를 출력하세요
        if skipgram:
            print([[tokens[center], tokens[t]] for t in surrounds])
            # print(*[[center, idx] for idx in surrounds])  # unpacking:별을 앞에다 붙이면 강제로 unpacking 이 된다.

        else:
            print([tokens[t] for t in extract(len(tokens), center, 2)],tokens[center] )
    
    return


tokens = 'The quick brown fox jumps over the lazy dog'.split()
show_word2vec(tokens, skipgram=True)
show_word2vec(tokens, skipgram=False)
#퀴즈3: cbow 방식도 skip-gram 처럼 토큰을 직접 출력하세요

