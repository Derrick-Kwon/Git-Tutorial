#CNN의 sliding도 시계열이다 따라서 CNN에서 배운 알고리즘을 RNN 에 적용해도 결과가 상당히 잘나온다
#cnn = converlutional neural network
#
def mnist_cnn():
    import tensorflow.keras as keras
    import numpy as np

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    x_train = x_train.reshape(-1, 28, 28, 1) #cnn 이기 때문에 3차원으로 바꿔준다.
    x_test = x_test.reshape(-1, 28, 28, 1)

    print(np.min(x_train), np.max(x_train))
    x_train = x_train / 255
    x_test = x_test / 255  # 가장 큰 값이 255 이므로!/ 표준편차의 오차값 차이를 줄여준다!


    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=x_train.shape[1:]))
    model.add(keras.layers.Conv2D(filters=6, kernel_size=[5, 5], strides=1, padding='same', activation='relu')) #슬라이딩을 2차원으로 하기 때문에! #same대신에 valid를 쓰면 크기가 줄어든다.
    model.add(keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same'))
    model.add(keras.layers.Conv2D(filters=6, kernel_size=[5, 5], strides=1, padding='same',activation='relu')) #변수 순서가 정해져있으므로 빨간색 부분은 지워도 된다. activaion func 빼고!
    model.add(keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same'))
    model.add(keras.layers.Conv2D(filters=6, kernel_size=[5, 5], strides=1, padding='same', activation='relu')) #conv 뒤에는 반드시!!! relu 가 들어가야 한다!
    model.add(keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')) #나중에 poolsize돌릴때 shape이 홀수면 padding을 추가해서 짝수로 만들거나 한칸을 버리거나 하는 방법을 사용하면 된다.
    #Dense 레이어는 2차원 데이터가 들어간다. but conv 는 4차원이 들어간다. 따라서 중간에 중간다리가 있어야 한다.
    # model.add(keras.layers.Reshape([-1]))
    model.add(keras.layers.Flatten()) #1차원데이터로 만들어준다.
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer=keras.optimizers.SGD(),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    return
mnist_cnn()