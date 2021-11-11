#먼저 Day_12_01_VGG16과 같이 모델을 만들고, summary로 한번 보라!
#인공지능은 추천 쪽이 제일 방대하고 할일이 많다.
#누군가의 코드분석할 때 순서 : view -> tool windows -> structure
#언젠가 좋은코드를 가져다 써야할 때가 있다. 그때 이렇게 모르는건 지우라
#keras.application 안에서 이것저것 갔다가 쓸 수 있다.
#functional에서 여러 브랜치(갈림)이 나오게되는데 어디서 갈라지는데 판단하기 위해서는 각 층의 '이름' 이 중요하다.


import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np

img_input = layers.Input(shape=[224, 224, 3])
# Block 1
x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# Block 3
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# Block 4
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

# Block 5
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

# if include_top: #dense를 top이라고 부른다(cnn 에서 전문용어로)

x = layers.Flatten(name='flatten')(x)
x = layers.Dense(4096, activation='relu')(x)
x = layers.Dense(4096, activation='relu')(x)
x = layers.Dense(1000, activation='softmax')(x)

model.summary()

model = keras.Model(img_inputs, x, name='vgg16')


