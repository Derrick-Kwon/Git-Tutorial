#Day_02_02_LogisticRegression  - classification / 둘 중 하나 택하기
#classification 에서는 loss func 으로 crossentropy 를 사용한다!!! (암기)
#evaluate [loss, accuracy] 반환 과 predict 함수(뭐가틀린거야??)
#evaluate 쓰는 대신  validation_data=[x_test, y_test] 를 epoch 옆에 써도 된다.
#acuract 값이 들쭉날쭉하면 일단 sgd 값 작게바꾸기
#데이터가 각각의 x 값의 단위가 다른 등, 잘못되었을경우 'scailing' 이 필요하다!!(중요)
##scikit-learn 모듈 설치 -
a = [1, 2, 3, 4, 5]
print(a[1:3])
def logistic_regression():
    import tensorflow.keras as keras
    import numpy as np
    #1 공부일수, 2출석일수

    x = [[1, 2],     #탈락
         [2, 1],
         [4, 5],     #통과
         [5, 4],
         [8, 9],
         [9, 8]]

    #0,1 구분은 decoder 부분에 따라서 의미가 달라진다.
    #계산에 log 가 들어가기 때문에 0,1 이외의 값은 상관없다.
    #퀴즈3 predict 함수를 사용해서 정확도를 직접 계산하시오!

    y = [[0],
         [0],
         [1],
         [1],
         [1],
         [1]]

    model = keras.Sequential()
    model.add(keras.layers.Dense(1, activation=keras.activations.sigmoid))
    model.compile(optimizer=keras.optimizers.SGD(),
                  loss=keras.losses.binary_crossentropy,
                  metrics=['accuracy'])
    model.fit(x, y, epochs=10)
    print(model.evaluate(x, y))   #[loss, accuracy] 형태로 나온다.
    print(model.predict(x))
    p = model.predict(x)

    p_bools=(p>0.5)
    print(p_bools)

    p_ints = np.int32(p_bools)    #이 코드 중요하다! np.int32 float 등 여러 배열을 단번에 int32 로 바꾼다.
    print(p_ints)

    equals = (p_ints == y)
    print(equals)

    print('acc :', np.mean(equals))

    #앞에 두친구가 계속 틀리네? -> 왜 틀리는가 분석!!-> 여기서 predict가 쓰이므로 중요하다!
    return


def logistic_regression_pima():
    #데이터 가져오기 당뇨병데이터 pima-indians-diabets
    #문제3: 70프로로 학습하고 30프로에 대해서 정확도를 구하시오!(정확도 75이상)
    import tensorflow.keras as keras
    import pandas as pd
    from sklearn import preprocessing #preprocessing중요하다!! - scailing, minmaxscailing 주로 씀
    import numpy as np
    pima_data = pd.read_csv('data/pima-indians-diabetes.csv', header=None) #헤더가 none 은 index 가 없다는거다.
    print(pima_data)

    x = pima_data.values[:, :-1]
    y = pima_data.values[:, -1:] #이런식으로 쓰면 2차원배열이 나온다!
    # print(x.shape, y.shape)
    x = preprocessing.scale(x) #scale은 x 만 적용/분산등을 이용/두개다 돌아가면서 적용해보고 좋은걸 쓰자!(중요)

    x_train = x[:500]; x_test = x[500:]; y_train= y[:500]; y_test = y[500:]

    model = keras.Sequential()
    model.add(keras.layers.Dense(1, activation=keras.activations.sigmoid))
    model.compile(optimizer=keras.optimizers.SGD(0.01),
                  loss=keras.losses.binary_crossentropy,
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
    print(model.evaluate(x_test, y_test))   #[loss, accuracy] 형태로 나온다.
    return
    print(model.predict(x))
    p = model.predict(x)

    p_bools=(p>0.5)
    print(p_bools)

    p_ints = np.int32(p_bools)
    print(p_ints)

    equals = (p_ints == y)
    print(equals)

    print('acc :', np.mean(equals))



logistic_regression_pima()

