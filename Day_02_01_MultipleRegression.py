#Day_02_01_MultipleRegression
#모두를 위한 딥러닝
#모든데이터는 다음 x, y 와 같이 나오게된다(이게 정규 format임)
#y데이터의 형태에 따라서 에 따라서 어떤 모델형태인지 판가름 할 수 있다
# 퀴즈0 : 아래 데이터에 대해서 모델을 구축하시오!
#맨처음에 shape error 가 났는데 이는, tupel에서는 shape 이 지원 안되기 때문!
# ^위 문제 해결위해선 항상 어떻게 데이터가 생겼는지 한번 봐보자!
#epochs 옆에 verbose 는 train 과정을 깔끔히 한다

def MultipleRegression():
     import tensorflow.keras as keras

     x = [[1, 0],
          [0, 2],
          [3, 0],
          [0, 4],
          [5, 0]]
     y = [[1],
          [2],
          [3],
          [4],
          [5]]

     model = keras.Sequential()
     model.add(keras.layers.Dense(1))
     model.compile(optimizer=keras.optimizers.SGD(), loss=keras.losses.mse)
     model.fit(x, y)
     return


# 전처리 중 x_train 과 y_train 의 차원을 맞춰주는게 습관이 되면 좋다.(중요)
# 여기서 차원 맞추기는
def MultipleRegression_boston():
     import tensorflow.keras as keras
     import numpy as np
     #퀴즈1 : 보스턴 집값 데이터에 포함된 학습과 검사 데이터의 shape 을 알려주세요
     boston_train, boston_test = keras.datasets.boston_housing.load_data(test_split=0.2)
     x_train, y_train = boston_train
     x_test, y_test = boston_test
     # print(type(boston_train))
     # print(x_train[:10])
     # print(y_train[:10])
     #퀴즈2 보스턴 집값데이터에 대해 80퍼의 데이터로 학습하고, 20퍼센트 데이터의 평균 오차를 구하시오


     model = keras.Sequential()
     model.add(keras.layers.Dense(1))
     model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.000001),
                   loss=keras.losses.mse,
                   metrics=['mae'])
     model.fit(x_train, y_train, epochs=10)
     p = model.predict(x_test)
     p = p.reshape(-1)
     e = p - y_test.reshape(-1)

     print('mae :', np.mean(np.absolute(e)))
     print('mse :', np.mean(e ** 2))
     return

print(MultipleRegression_boston())

