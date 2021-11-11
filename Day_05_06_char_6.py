#Day06_rnn #자연어처리 옛날꺼 nltk 최신 RNN
#정규표현식 다시 공부할 것!
#구글에 drum up the people 검색
import nltk
import numpy as np
import Day_05_05_Char_rnn_5
def show_ngrams():
    tokens = 'i work at intel'.split()
    # print(list(nltk.bigrams(tokens))#grams 는 단위다, bi는 2개씩 묶는거 #generator 은 for에서 돌릴수 있는 객체를 뜻한다
    print(list(nltk.ngrams(tokens, 2)))
    w = 'tensor'
    print(list(nltk.ngrams(w, 2)))

long_sentence = ("If you want to build a ship,"
 "don't drum up people to collect wood and don't assign them tasks and work,"
  "but rather teach them to long for the endless immensity of the sea.")
#기행문자는 안쓰는게 좋다!
#
# sequence_sentence = []
# for i in range(20, len(long_sentence)+1):
#     sequence_sentence.append(long_sentence[i-20:i])

#강사님 코드
seq_length = 20
words = nltk.ngrams(long_sentence, seq_length)
# x = [w[:-1] for w in words] #주의tip! 제너레이터 : words에 대해서 이미 한번 for을 사용해서 y에대해서는 작동하지 않는다!
# y = [w[1:] for w in words]
print(list(words)[0])
# print(np.array(x).shape, np.array(y).shape) #문자열이기때문에 array를 써줘야 한다



#퀴즈5-6-1 시퀸스 길이 20개로 문장을 나누세요.
#20개 안에는 x가 19개, y가 19개 들어 있습니다.
#x, y끼리 서로서로 모을때 ngrams 를 써보자!

words = [''.join(w) for w in words]
Day_05_05_Char_rnn_5.char_rnn_5(words)

x = nltk.ngrams(long_sentence[:-1], seq_length-1)
y = nltk.ngrams(long_sentence[1:], seq_length-1)
print(list(x), np.array(y).shape)
