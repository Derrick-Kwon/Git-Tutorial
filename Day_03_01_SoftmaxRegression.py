#Day_03_03
#ctrl + shift+ f 전체코드/함수 찾기
import tensorflow.keras as keras
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection


# 퀴즈1: y데이터 만들기
# 퀴즈2: 예측 결과의 합계가 1이 되는지 증명하세요
def softmax_regression():

    x = [[1, 2],     #a학점
         [2, 1],
         [4, 5],     #b학점
         [5, 4],
         [8, 9],     #a학점
         [9, 8]]



    # 계산에 log 가 들어가기 때문에 0,1 이외의 값은 상관없다.
    #클래스가 3개이므로
    y = [[0, 0, 1],
         [0, 0, 1],
         [0, 1, 0],
         [0, 1, 0],
         [1, 0, 0],
         [1, 0, 0]]

    model = keras.Sequential()
    model.add(keras.layers.Dense(3, activation=keras.activations.softmax)) #softmax 는 전체합 1이다. /soft는 많은 클래스중 비중을 구할때 쓴다.
    model.compile(optimizer=keras.optimizers.SGD(),
                  loss=keras.losses.categorical_crossentropy, #categorical 은 y 값이 원-핫벡터란 뜻
                  metrics=['accuracy'])
    model.fit(x, y, epochs=10)
    print(model.predict(x))
    p = model.predict(x)
    # for i in p:
    #     print(np.sum(i))
    print(np.sum(p, axis = 1)) #0(수직), 1(수평)
    print('-'*30)
    p_arg = np.argmax(p, axis=1)
    y_arg = np.argmax(y, axis=1)
    print(p_arg)
    print(y_arg)
    print('acc: ', np.mean(p_arg == y_arg))

    #문제2: argmax 함수를 이용하여 정확도를 구하시오!
    return


def softmax_regression_iris():
#iris.csv 복붙해서
#iris.csv 70&로 학습하고 30%에 대해 정확도를 구하시오

    iris_data = pd.read_csv('data/iris_onehot.csv', index_col=0, skiprows =0)
    print(iris_data)
    values1 = iris_data.values
    np.random.shuffle(values1)

    x = values1[:, :-3]
    y = values1[:, -3:]    #csv를 가져온 dataFrame 형태에서는 iris_data.values[???] 형태이지만,
                           #이미 위에서 values를 가져왔으므로 상관없다!
    # print(x.shape, y.shape)
    # print(x[0:4], y[0:3])

    train_size = int(len(x)*0.7)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # y_train = y_train1 + y_train2 + y_train3
    # x_test = x_test1 + x_test2 + x_test3
    # y_test = y_test1 + y_test2 + y_test3
    # print(x_train[:4])
    # 위 함수 대신에 shuffle 함수 이용해보자!


    model = keras.Sequential()
    model.add(keras.layers.Dense(3, activation=keras.activations.softmax)) #softmax 는 전체합 1이다. /soft는 많은 클래스중 비중을 구할때 쓴다.
    model.compile(optimizer=keras.optimizers.SGD(),
                  loss=keras.losses.categorical_crossentropy, #categorical 은 y 값이 원-핫벡터란 뜻
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10)
    print(model.predict(x))
    print(model.evaluate(x, y))

    return


def softmax_regression_iris_dense():
#iris.csv 복붙해서
#iris.csv 70&로 학습하고 30%에 대해 정확도를 구하시오

    iris_data = pd.read_csv('data/iris.csv', index_col=0, skiprows =0)
    print(iris_data)


    x = iris_data.values[:, :-1]
    y = iris_data.values[:, -1:]    #csv를 가져온 dataFrame 형태에서는 iris_data.values[???] 형태이지만,
                           #이미 위에서 values를 가져왔으므로 상관없다!
    print(x.shape, y.shape)
    print(x.dtype, y.dtype )#넘파이는 string 지원 x => 문자, 숫자 통틀어서 object로 출력!

    #values 를 먼저 가지고 온후 [ :] 트래이싱 발생하므로 전부 object형태로 저장된다!
    x = np.float32(x) #따라서 이렇게 형태를 바꿔줘야 한다!

    enc = preprocessing.LabelBinarizer()
    y = enc.fit_transform(y)
    print(y) #y전처리 원-핫 인코딩으로 코드 (***중요!!!!!!)

    # train_size = int(len(x)*0.7)
    # x_train, x_test = x[:train_size], x[train_size:]
    # y_train, y_test = y[:train_size], y[train_size:]
    #위 #3개 코드 대신 model_selection사용할 수 있다!

    data = model_selection.train_test_split(x, y, train_size = 0.7)
    x_train, x_test, y_train, y_test = data



    # y_train = y_train1 + y_train2 + y_train3
    # x_test = x_test1 + x_test2 + x_test3
    # y_test = y_test1 + y_test2 + y_test3
    # print(x_train[:4])
    # 위 함수 대신에 shuffle 함수 이용해보자!


    model = keras.Sequential()
    model.add(keras.layers.Dense(3, activation=keras.activations.softmax)) #softmax 는 전체합 1이다. /soft는 많은 클래스중 비중을 구할때 쓴다.
    model.compile(optimizer=keras.optimizers.SGD(),
                  loss=keras.losses.categorical_crossentropy, #categorical 은 y 값이 원-핫벡터란 뜻
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10)
    print(model.predict(x))
    print(model.evaluate(x, y))

    return


def softmax_regression_iris_sparse():
    #원핫인코딩은 편하지만 가독성이 떨어진다  y데이터 : 000000100000000 -> 7 (sparse data)이런식으로
    iris_data = pd.read_csv('data/iris.csv', index_col=0, skiprows =0)

    x = iris_data.values[:, :-1]
    y = iris_data.values[:, -1:]
    x = np.float32(x)

    # enc = preprocessing.LabelBinarizer()
    print(y)
    data = model_selection.train_test_split(x, y, train_size = 0.7) #여기서 랜덤으로 뽑아주는 것도 있다.
    x_train, x_test, y_train, y_test = data

    model = keras.Sequential()
    model.add(keras.layers.Dense(3, activation=keras.activations.softmax)) #softmax 는 전체합 1이다. /soft는 많은 클래스중 비중을 구할때 쓴다.
    model.compile(optimizer=keras.optimizers.SGD(),
                  loss=keras.losses.sparse_categorical_crossentropy, #categorical 은 y 값이 원-핫벡터란 뜻
                  metrics=['accuracy']) #loss를 계산할때의 y값 형태가 달라졌으므로 loss 를 sparse_ .... 로 바꾼다
    model.fit(x_train, y_train, epochs=10, verbose=0)
    print(model.evaluate(x, y))
    p = model.evaluate(x, y)
    # 퀴즈3: 정확도를 직접 계산하세요
    p_arg = np.argmax(p, axis =1)
    print('acc: ', np.mean(p_arg == y_test))  # True, False 의 평균을 낸다!!
    return


softmax_regression_iris_sparse()



#***중요!!!!
# 1. 멀티플, 로지스틱 / 소프트맥스 - (원핫인코딩, 라벨)
# 2. scailing, min,maxscailing
# 3. split 하는거