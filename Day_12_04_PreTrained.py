#Day_12_04_PreTrainied
#사전학습모델은 cnn만 적용이 된다
#
#flowers5 -- train -- buttercup 폴더
                 # -- coltsfoot 폴더
                 # -- daffodill 폴더
          # -- test -- buttercup 폴더
                 # -- coltsfoot 폴더
                 # -- daffodill 폴더
#이렇게 구조를 만들어놓으면 레이블(이름)을 하나하나 바꿔줄 필요가 없다!!!
#xtrain이 60개씩이 맞는지 꼭 세보는게 좋다.

import keras.preprocessing.image
from tensorflow.keras import layers
import tensorflow.keras as keras


gen_train = keras.preprocessing.image.ImageDataGenerator(rescale=1/255,#scailing 은 각각의 피쳐가 균등하게 만든다
                                                         zoom_range=1.5) #ctrl써서 들어가보면 그 참고가 있다.
                                                                         #그거 읽어보면서 데이터 증강을 할 수 있다.
                                                                        #preprocessing function등등...

flow_train = gen_train.flow_from_directory('flowers3/train',  #ctrl써서 들어가본다. 그리고 안에 필요한거를 채운다.
                                           target_size=[224, 224],
                                           class_mode='sparse') #categorical, binary, sparse' 기본 코드 xx

get_test = keras.preprocessing.image.ImageDataGenerator()
flow_test = get_test.flow_from_directory('flowers3/test',
                                         target_size=[224, 224],
                                         class_mode='sparse')
# exit() #여기서 쓰면 잘 불러왔는지 아닌지 알 수 있다!!

conv_base = keras.applications.VGG16(include_top = False, input_shape=[224, 224, 3]) #top은 VGG16에서 덴스층을 말한다!(우리껄 써야하므로 False)
conv_base.trainable = False #그 vGG16안에 있는 weight를 학습하지 않겠다는 것이다.(처음부터 하겠다는 뜻- 이거로 고정!)
model = keras.Sequential()

#Dense 레이어는 2차원 데이터가 들어간다. but conv 는 4차원이 들어간다. 따라서 중간에 중간다리가 있어야 한다.
# model.add(keras.layers.Reshape([-1]))
model.add(conv_base)
model.add(keras.layers.Flatten()) #4차원 -> 2차원
model.add(keras.layers.Dense(4096, activation='relu'))
model.add(keras.layers.Dense(4096, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))
model.summary()

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
model.summary()
model.fit(flow_train, epochs=5, batch_size=16, verbose=2)
#model.fit_generator은 안쓴다!


