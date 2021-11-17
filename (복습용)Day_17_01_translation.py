#Day_17_01_translation
#캐릭터 단위의 번역모델을 만들 것이다!(단어넣음)
#그다음엔 단어 단위의 번역모델(문장넣음)
#sequence 2 sequence 모델이란? encoder의 학습한 것으로 decoder의 첫단어를 예측하는데 쓴다
#이렇게 학습한 각각의 weight는 context(문맥)이 되고, 이를 decoder로 넘긴다.

#def make_xy 만드는부분 복습!!!!



#퀴즈1
#토큰 사전을 만들고, 토큰 인덱스(idx2chr, 딕셔너리)를 반환하는 함수를 만드세요
import numpy as np
def make_vocab(data):
        '''
        a = []
        #내코드


        for i in data:
                a.append(i[0])
        join_a = list(set(''.join(a)))
        join_a = sorted(join_a)
        # print(join_a)
        add = []
        for idx, el in enumerate(join_a):
                add.append([el, idx])
        vocab = dict(add) #dict는 2차원배열을 -> 순서쌍으로 바꿔주는것이다!
        '''

        #강사님 코드
        eng = sorted(set(''.join([e for e, _ in data])))
        kor = sorted(set(''.join([k for e, k in data])))

        vocab = eng + kor + list('SEP') #start, End, pad(시작과 끝에 넣는건데 일단 끝에만)

        chr2idx= {v:i for i, v in enumerate(vocab)}
        return vocab, chr2idx



def make_xy(data, chr2idx):
        onehots = np.eye(len(chr2idx), dtype=np.int32)

        enc_x, dec_x, dec_y = [], [], []
        for e, k in data:
                enc_in = [chr2idx[c] for c in e]
                dec_in = [chr2idx[c] for c in 'S'+k]
                target = [chr2idx[c] for c in k+'E']
        print(enc_in, dec_in, target)
        enc_x.append(onehots[enc_in])
        dec_x.append(onehots[dec_in])
        dec_y.append(target)
        print(enc_x[-1])

        #퀴즈: enc_in 을 원핫 벡터로 변환해서 추가하세요
        return np.int32(enc_x), np.int32(dec_x), np.int32(dec_y)










data = [('food', '음식'), ('pink', '분홍'),
        ('wind', '바람'), ('desk', '책상'),
        ('head', '머리'), ('hero', '영웅')]

vocab, chr2idx = make_vocab(data)
enc_x, dec_x, dec_y = make_xy(data, chr2idx)