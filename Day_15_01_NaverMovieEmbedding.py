#Day_15_02_NaverMovieEmbedding
#퀴즈1 이전파일의 원 핫 벡터를 임베딩 레이어로 교체


import re
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np


def get_xy(file_path):
    f = open(file_path, 'r', encoding='utf-8')

    # skip header
    f.readline()

    x, y = [], []
    for line in f:
        # print(line.strip().split('\t'))
        _, doc, label = line.strip().split('\t')

        x.append(clean_str(doc).split())
        y.append(int(label))

    f.close()
    # small = int(len(x) * 0.1)
    # return x[:small], np.int32(y[:small])

    return x, np.int32(y)


def clean_str(string):
    string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def save_model():
    x_train, y_train = get_xy('data/ratings_train.txt')
    x_test, y_test = get_xy('data/ratings_test.txt')

    vocab_size = 2000
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x_train)

    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_test_seq = tokenizer.texts_to_sequences(x_test)

    max_len = 35
    x_train_pad = keras.preprocessing.sequence.pad_sequences(x_train_seq, maxlen=max_len)
    x_test_pad = keras.preprocessing.sequence.pad_sequences(x_test_seq, maxlen=max_len)


    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=x_train_pad.shape[1:]))    #input에 1차원 데이터 전달됨!
    model.add(keras.layers.Embedding(vocab_size, 100)) #낑겨넣기! (아무리 길어도 300~500로 만듬) #2차원이 들어와서 3차원으로 만들어 lstm에 전달한다
    model.add(keras.layers.LSTM(50))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.binary_crossentropy,
                  metrics='acc')

    checkpoint = keras.callbacks.ModelCheckpoint(
        'model/naver_{epoch:02d}_{val_loss:.2f}.h5',
        save_best_only=True)

    model.fit(x_train_pad, y_train, epochs=2, batch_size=64, verbose=2,
              validation_data=(x_test_pad, y_test),
              callbacks=checkpoint)
    model.save('model/embedding.h5')

def load_model():
    model = keras.models.load_model('model/embedding.h5')

    x_train, y_train = get_xy('data/ratings_train.txt')

    vocab_size, max_len = 2000, 35
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x_train)

    tokens = ['뭐야', '이', '평점들은', '나쁘진', '않지만', '10점', '짜리는', '더더욱', '아니잖아']
    for i in range(1, len(tokens) + 1):
        x_review = tokens[:i]
        x_review_seq = tokenizer.texts_to_sequences(x_review)
        x_review_pad = keras.preprocessing.sequence.pad_sequences(x_review_seq, maxlen=max_len)

        p = model.predict(x_review_pad)
        print(p[0, 0], ' '.join(tokens[:i]))


# save_model()
load_model()
