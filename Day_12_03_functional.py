#Day_12_03_functional.py
import tensorflow.keras as keras
import numpy as np
# 회귀모델 = (멀티플regression, 로지스틱regression(정답이 0과1일때) 소프트맥스 regression)
#AND연산

def and_sequential():
    #AND
    data = [[0,0,0],
            [0,1,0],
            [1,0,0],
            [1,1,1]]
    data = np.float32(data)

    x = data[:, :-1]
    y = data[:, -1]

    print(x, y, x.shape)

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=x.shape[1:])) #전체를 집어넣으면 안되고 하나씩꺼내서 넣는다고 생각하면 된다!
    model.add(keras.layers.Dense(1))
    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.mse)
    model.fit(x,y, epochs=100, validation_data=(x, y))
    print(model.predict([[0, 0]]))
    print(model.evaluate(x, y))


def and_functional():
    # AND
    data = [[0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 1]]
    data = np.float32(data)

    x = data[:, :-1]
    y = data[:, -1]



    input = keras.layers.Input(shape=x.shape[1:]) #얘네 각각은 변수이다.
    dense = keras.layers.Dense(1, activation='sigmoid')

    output = dense.__call__(input) #이 친구는 dense(input) 이거랑 같은 코드이다. 여기서 가로 안에 있는것을 함수호출연산자라고 한다.



    output = keras.layers.d(1, activaion='sigmoid')(input) #따라서 이렇게 고치면 작동한다!



    model = keras.Model(input, output)

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.mse)
    model.fit(x, y, epochs=100, validation_data=(x, y))
    print(model.predict([[0, 0]]))
