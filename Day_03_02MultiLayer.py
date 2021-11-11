#Day_03_02_MultiLayer_Mnist!
def minst_softmax():
    import tensorflow.keras as keras
    import numpy as np
    #퀴즈0 mnist 데이터의 shape 을 출력하시오
    (x_train, y_train), (x_test, y_test)= keras.datasets.mnist.load_data()
    #튜플 형태이므로 다음과 같이 가져와본다!
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    #퀴즈1: mnist 데이터 셋에 대해 동작하는 모델을 구축하세요


    x_train = x_train.reshape(60000, 784)
    x_test =x_test.reshape(10000, 784)

    print(np.min(x_train), np.max(x_train))
    x_train = x_train/255
    x_test = x_test/255  #가장 큰 값이 255 이므로!/ 표준편차의 오차값 차이를 줄여준다!

    model = keras.Sequential()
    model.add(keras.layers.Dense(10, activation=keras.activations.softmax))  # softmax 는 전체합 1이다. /soft는 많은 클래스중 비중을 구할때 쓴다.
    model.compile(optimizer=keras.optimizers.SGD(),
                  loss=keras.losses.sparse_categorical_crossentropy,  # categorical 은 y 값이 원-핫벡터란 뜻
                  metrics=['accuracy'])  # loss를 계산할때의 y값 형태가 달라졌으므로 loss 를 sparse_ .... 로 바꾼다
    model.fit(x_train, y_train, epochs=10, verbose=0)
    print(model.evaluate(x_test, y_test))


def minst_multi_layers():
    import tensorflow.keras as keras
    import numpy as np

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    print(np.min(x_train), np.max(x_train))
    x_train = x_train / 255
    x_test = x_test / 255  # 가장 큰 값이 255 이므로!/ 표준편차의 오차값 차이를 줄여준다!

    model = keras.Sequential()
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    #멀티 레이어는 non-linear 형태여야 한다.
    #레이어를 단절시키는것이 정말정말! 중요하다(vanishing gradient가 어떤 뜻인지 문제질문하기!)
    model.compile(optimizer=keras.optimizers.SGD(),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10)
    print(model.evaluate(x_test, y_test))

    return

minst_multi_layers()
#그 맞는 함수를 선택하는 과정! 을 고르자
# 1. y 값에 따라서 달라짐
# 2. y값이 2개     - logistic_Regression /
#    y값이 여러개   -