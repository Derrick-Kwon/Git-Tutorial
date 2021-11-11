

#ctrl + shift + f10
#alt + 1 / 옆에 창 열고닫기
#alt + 4 / 밑에 창 열고닫기
#tab /오른쪽으로 밀기,  shift + tab / 왼쪽으로 밀기
#ctrl + /  일괄 주석처리
#pip 로 설치하는 것과는 조금 다른가?
#epochs 를 n 번해도 loss 가 계속 떨어지면 더 돌려도 된다는 뜻이다.
#------------------------------------------------------------------------
# read_csv / cars.values 로 따로 필요한거 가져올수 있다 - numpy 배열로!
#    x = cars.values[:, :-1] 이 슬라이싱 문법 중요하다! (필요한거 가져오기)
#192.168.0.48 강사님 ip 서버
import numpy as np
import pandas as pd

print("Hello World")

#퀴즈0
#아래 데이터에 대해 동작하는 케라스 모델 구축하세요


def linear_regression():
    import tensorflow.keras as keras

    x = [1, 2, 3]
    y = [1, 2, 3]

    model = keras.Sequential()
    model.add(keras.layers.Dense(1))
    model.compile(optimizer=keras.optimizers.SGD(), loss=keras.losses.mse)
    model.fit(x, y, epochs=100)
    p = model.predict(x)
    p = p.reshape(-1)
    e = p - y
    print(e)

    print('mae :', np.mean(np.absolute(e)))
    print('mse :', np.mean(e ** 2))

#퀴즈1 : 속도가 30과 50 일때의 제동 거리를 구하시오
#퀴즈2 : 구축한 모델을 시각화 하시오


def linear_regression_cars():
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow.keras as keras
    cars = pd.read_csv('data/cars.csv', index_col=0)
    #print(cars.values)
    x = cars.values[:, 0]
    y = cars.values[:, 1]
    #print(x.shape, y.shape)  #(50,) (50,)

    model = keras.Sequential()
    model.add(keras.layers.Dense(1))
    model.compile(optimizer=keras.optimizers.SGD(0.0001),
                  loss=keras.losses.mse
                  )
    model.fit(x, y, epochs=100)
    p, p1, p2 = model.predict([0, 30, 50])
    z = z.reshape(-1)

    plt.plot(x, y, 're')
    plt.plot([0, 30], [0, z], 'g')
    plt.plot([0, 30], [z, z], 'b')
    plt.show()
    return


print(linear_regression())




