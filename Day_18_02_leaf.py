#Day_18_02_leaf.py
#퀴즈
#캐글에서 나뭇잎 경진대회의 모델을 만들어서 결과를 제출하세요
#캐글에서 꼼수로 제출 한번 해보기!

#공부순서
# 1. 수업시간에 배웠던것 다시 해보기!
# 2. 텐서플로 자격증
# 3. 새로운 문제 uci machine-learning 가서 popular_data 한번씩 다 풀어보기
# 4. 모두를 위한 딥러닝 시즌1

import pandas as pd
import numpy as np
import tensorflow.keras as keras
from sklearn import preprocessing
import os

def get_train():
    leaf = pd.read_csv('leaf-classification/train.csv', index_col=0)
    # print(leaf)

    x = leaf.values[:, 1:]

    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(leaf.species)

    return np.float32(x), y, enc.classes_


def get_test():
    leaf = pd.read_csv('leaf-classification/test.csv', index_col=0)
    # print(leaf)

    return np.float32(leaf.values), np.int32(leaf.index.values)


def make_submission(user_ids, predictions, filename):
    f = open(os.path.join('model', filename), 'w', encoding='utf-8')

    f.write('"id","sentiment"\n')
    for uid, p in zip(user_ids, predictions):
        # print(uid, int(p > 0.5))
        f.write('"{}",{}\n'.format(uid, int(p > 0.5)))

    f.close()



x_train, y_train, classes = get_train()
x_test, leaf_ids = get_test()

print(x_train.shape, y_train.shape)     # (990, 192) (990,)
print(y_train[:5])                      # [ 3 49 65 94 84]


print(x_test.shape, leaf_ids.shape)     # (594, 192) (594,)
print(leaf_ids[:5])                     # [ 4  7  9 12 13]

model = keras.Sequential()
model.add(keras.layers.Dense(len(classes), activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics='acc')

model.fit(x_train, y_train, epochs=10, verbose=2, validation_split=0.2)

p = model.predict(x_test)
p_arg = np.argmax(p, axis=1)

eye = np.eye(len(classes), dtype=np.float32)


