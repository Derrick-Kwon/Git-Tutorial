#Day_10_02_temperature

import numpy as np

def sigmoid(z):
    return 1 / (1 + np.e**-z)

def softmax_1(z):
    s = np.sum(z)
    return z/s

def softmax_2(z):
    z =np.exp(z)                           #np.e ** z
    s = np.sum(z)
    return z/s

def temperature(z, t):
    z = np.log(z)/t
    z = np.exp(z)
    s = np.sum(z)
    return z/s

def weighted_pick(p):
    t = np.cumsum(p)
    # print(t) #[0.3 0.5 0.9 1. ]
    print(np.searchsorted(t, 0.6)) #그 뒤에 변수가 들어갔을때 넣고 자동정렬!

    n = np.random.rand(1)[0] # 0~1사이의 균등분포(다 같은 확률로 뽑아냄) (정규분포랑 다르다!)
    print(np.seqrchsorted(t, n), n)


a = [2.0, 1.0, 0.1]
a = np.float32(a)
# 하나씩 더한다는 개념이 리스트엔 없다 따라서 numpy로 변경 -> (브로드캐스트기능 넣는다)

print(np.e) #2.718281828459045
print(sigmoid(a)) #[0.880797  0.7310586 0.5249792]
print(softmax_1(a)) #[0.64516133 0.32258067 0.03225807]
print(softmax_2(a)) #[0.6590011  0.24243295 0.09856589]
print(temperature(a, 0.1))
print(temperature(a, 0.5)) [0.79840314 0.19960079 0.00199601]
print(temperature(a, 0.8)) [0.69247675 0.29115063 0.0163726 ]

print(weighted_pick([0.3, 0.2, 0.4, 0.1])) #argmax는 무조건 0.4인데 반해서, 각각뽑힐 확률이 0.3 0.2 0.4 0.1이다


이 템퍼레쳐에 따라서 모델이 선택하는 단어가 달라진다!!
따라서 다양한 소설을 만들 수 있게 된다. 