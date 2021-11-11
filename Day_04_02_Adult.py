#Day_04_02_adult.py
import tensorflow.keras as keras
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
#채소 공급시스템? 채소 체인점

#퀴즈 :1단계 파일 읽기
# age: continuous.
# workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# fnlwgt: continuous.
# education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# education-num: continuous.
# marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# sex: Female, Male.
# capital-gain: continuous.
# capital-loss: continuous.
# hours-per-week: continuous.
# native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
def get_data_encoder(file_path):
    names = ['age','workclass','fnlwgt','education',
             'education-num','marital-status','occupation',
             'relationship','race','sex','capital-gain','capital-loss',
             'hours-per-week','native-country',
             'income']

    train_data = pd.read_csv(file_path, names= names,#이런식으로 설명하는 부분을 만들 수 있다!!
                             sep=', ', #애들이 ', ' 로 구분해놔서, 구분자를 만들어준다.
                             engine='python')
    print(train_data.info()) #이걸 이용해서 각 줄의 데이터타입을 본다
    #우선 integer 만 모은다

    x = [train_data['age'].values, train_data['fnlwgt'].values, train_data['education-num'].values,
         train_data['capital-gain'].values, train_data['capital-loss'].values,
         train_data['hours-per-week'].values]
    # 퀴즈3 : categorical: 문자열 데이터를 x 데이터에 추가하세요! (2가지방법이 있다)
    x = np.int32(x) #처음 name로 가져올 때 생각해보면, 문자 섞여있기 때문에 바꿔줘야 한다
    x = np.transpose(x) #그 이전 파일들 처럼 밑으로 쭉 나열되게 한다!
    x = preprocessing.scale(x)  # 스케일링은 무조건 넣어준다! #minmaxscaling 도 넣어보자!

    # 이번엔 y를 바꿔준다!
    enc = preprocessing.LabelBinarizer()
    y = enc.fit_transform(train_data['income'].values)
    # print(y)

    #다른 카테고리도 바꿔준다.(강사님 코드참고)
    workclass = enc.fit_transform(train_data['workclass'].values)
    education = enc.fit_transform(train_data['education'].values)
    marital = enc.fit_transform(train_data['marital-status'].values)
    occupation = enc.fit_transform(train_data['occupation'].values)
    relationship = enc.fit_transform(train_data['relationship'].values)
    race = enc.fit_transform(train_data['race'].values)
    sex = enc.fit_transform(train_data['sex'].values)
    native = enc.fit_transform(train_data['native-country'].values)

    #마지막 퀴즈: binarizer여기서 native 와 sex 를 처리할때 클래스 갯수가 달라서 스킵!

    added = [workclass, education, marital, occupation,
             relationship, race, sex, native]
    added = np.transpose(added)
    added = preprocessing.scale(added)

    #여기 중요***

    x = np.concatenate([x, added], axis=1)
    # print(x.shape) #(32561, 14)
    return x, y


def get_data_binarizer(file_path):    #특정데이터에 대해서는 원핫벡터가 성능이 좋다 그러나, 다 바꾸기엔 시간이 너무 많이 들으므로 선별적으로 하는게 좋다
    names = ['age', 'workclass', 'fnlwgt', 'education',
             'education-num', 'marital-status', 'occupation',
             'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
             'hours-per-week', 'native-country',
             'income']

    train_data = pd.read_csv(file_path, names=names,  # 이런식으로 설명하는 부분을 만들 수 있다!!
                             sep=', ',  # 애들이 ', ' 로 구분해놔서, 구분자를 만들어준다.
                             engine='python')
    # print(train_data)

    x = train_data.values[:, :-1]
    y = train_data.values[:, -1]

    print(train_data['age'].values)  # values 함수는 실제 데이터만 가져온다

    print(train_data.info())  # 이걸 이용해서 각 줄의 데이터타입을 본다
    # 우선 integer 만 모은다

    print(a)
    x = [train_data['age'].values, train_data['fnlwgt'].values, train_data['education-num'].values,
         train_data['capital-gain'].values, train_data['capital-loss'].values,
         train_data['hours-per-week'].values]
    # 퀴즈3 : categorical: 문자열 데이터를 x 데이터에 추가하세요! (2가지방법이 있다)
    x = np.transpose(x)  # 그 이전 파일들 처럼 밑으로 쭉 나열되게 한다!

    # 다른 카테고리도 바꿔준다.(강사님 코드참고)
    workclass = enc.fit_transform(train_data['workclass'].values)
    added = np.transpose(workclass)

    # 이번엔 y를 바꿔준다!
    x = preprocessing.scale(x)  # 스케일링은 무조건 넣어준다! #minmaxscaling 도 넣어보자!
    enc = preprocessing.LabelBinarizer()
    y = enc.fit_transform(train_data['income'].values)
    # print(y)

    return x, y

#숙제로 binarizer 해볼 것!

def main():
    #x_test 가 똑같은 형식으로 하나 더 있기 때문에 함수형태로 만든다.
    x_train, y_train = get_data_encoder(('data/adult.data'))
    x_test, y_test = get_data_encoder(('data/adult.test'))

    #데이터 중간중간에 ? 가 있는데 이것을 처리하는 방법이 또 하나가 있다
    #근데 나중에 하겠다.

    print(x_train.shape)
    print(y_train.shape)

    #퀴즈2 : 앞에서 읽어온 데이터에 대해서 모델을 구축하세요.
    model = keras.Sequential()
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(0.01),
                  loss=keras.losses.binary_crossentropy,
                  metrics=['Accuracy'])
    model.fit(x_train, y_train, epochs=10,
              validation_data=(x_test, y_test),
              batch_size=32)
    return
main()