# Day_14_02_NaverMovie.py
import re
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# tsv 데이터는 tab 으로 데이터를 구분한다는 뜻이고, csv 로 읽을수 있다!!!
# 캐글에서 이런저런 (유명한) 문제 풀어봤다고 하면 취업에 +가 될 수 있다.
#이거 다하면 model파일에 csv 가 결과로 나오게 된다!!! 
def get_train(file_path):
    f = open(file_path, 'r', encoding='utf-8')

    # skip header
    f.readline()

    x, y = [], []
    for line in f:
        # print(line.strip().split('\t'))
        _, label, document = line.strip().split('\t')

        x.append(clean_str(document).split())
        y.append(int(label))

    f.close()
    return x, np.int32(y)

def get_test():
    df_test = pd.read_csv('word2vec-nlp-tutorial/testData.tsv', delimiter='\t', index_col=0)
    # print(df_test)

    x = [clean_str(r).split() for r in df_test['review']]
    y = df_test.index.values

    return x, y
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
    x_train, y_train = get_train('word2vec-nlp-tutorial/labeledTrainData.tsv')
    x_test, user_ids = get_test()
    print(x_test, user_ids)


    vocab_size = 2000
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x_train)


    x_train_seq = tokenizer.texts_to_sequences(x_train)

    # exit()
    # x_test_seq = tokenizer.texts_to_sequences(x_test)

    max_len = 400

    x_train_pad = keras.preprocessing.sequence.pad_sequences(x_train_seq, maxlen=max_len)
    # x_test_pad = keras.preprocessing.sequence.pad_sequences(x_test_seq, maxlen=max_len)

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=x_train_pad.shape[1:]))    # [max_len, vocab_size]
    model.add(keras.layers.Embedding(vocab_size, 100))
    model.add(keras.layers.LSTM(50))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.binary_crossentropy,
                  metrics='acc')

    checkpoint = keras.callbacks.ModelCheckpoint('model/popcorn_{epoch:02d}-{val_loss:.2f}.h5',
                                                 save_best_only=True) #이거 true로 주는거 중요하다!


    model.fit(x_train_pad, y_train, epochs=10, batch_size=64, verbose=2,
              validation_split=0.2,
              callbacks=checkpoint)

    model.save('model/popcorn.h5')

    return
def make_submission(user_ids, predictions, filename):
    # p = []
    # for i in predictions:
    #     p.append(int(i>0.5))
    #
    #
    # sub_dict = dict(zip(user_ids,p))
    # print(sub_dict)
    #
    # print(sub_dict['1736_10'])
    #
    # names = ["id", "sentiment"]
    # sub_csv1 = pd.read_csv('word2vec-nlp-tutorial/sampleSubmission.csv', names= names)
    # sub_csv2 = pd.read_csv('word2vec-nlp-tutorial/sampleSubmission.csv', names=names)
    # line = sub_csv1.read().replace(sub_csv1["id"], sub_dict["id"])
    # sub_csv2.write(line)

    f = open(os.path.join('model', filename), 'w', encoding='utf-8')

    f.write('"id","sentiment"\n')
    for uid, p in zip(user_ids, predictions):
        # print(uid, int(p > 0.5))
        f.write('"{}",{}\n'.format(uid, int(p > 0.5)))

    f.close()
    return


def load_model(model_name, submission_name):
    x_train, y_train = get_train('word2vec-nlp-tutorial/labeledTrainData.tsv')
    x_test, user_ids = get_test()
    # print(x_test, user_ids)


    vocab_size = 2000
    max_len = 400

    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x_train)

    x_test_seq = tokenizer.texts_to_sequences(x_test)
    x_test_pad = keras.preprocessing.sequence.pad_sequences(x_test_seq, maxlen=max_len)

    model = keras.models.load_model(model_name)
    p = model.predict(x_test_pad)
    # print(p.shape) (25000, 1)
    p = p.reshape(-1)   #(25000, )
    make_submission(user_ids, p, submission_name)

    return

# save_model()
load_model('model/popcorn_03-0.32.h5', 'popcorn_rnn.csv')