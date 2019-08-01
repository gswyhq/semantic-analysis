# -*- coding:utf-8 -*-
########################################
## import packages
########################################

# 来源 https://www.jianshu.com/p/a649b568e8fa

import os
import pickle

import keras.preprocessing.text
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from config import lstm_bst_model_path as bst_model_path, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, num_lstm, rate_drop_dense, rate_drop_lstm, act, \
    num_dense
from word2vector.embedding_matrix import generator_embedding_matrix, text_to_wordlist, text_to_word_sequence

########################################
# set directories and parameters
########################################

# ########################################
# ## index word vectors
# ########################################
# print('Indexing word vectors')
#
# word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
# print('Found %s word vectors of word2vec' % len(word2vec.vocab))

########################################
# process texts in datasets
########################################
print('Processing text dataset')

# print texts_1

'''
this part is solve keras.preprocessing.text can not process unicode
start here
'''

keras.preprocessing.text.text_to_word_sequence = text_to_word_sequence

########################################
# define the model structure
########################################
def get_model(nb_words, embedding_matrix):
    """
    网络结构
    该神经网络采用简单的单层LSTM+全连接层对数据进行训练，网络结构图
    该部分首先定义embedding_layer作为输入层和LSTM层的映射层，将输入的句子编码映射为词向量列表作为LSTM层的输入。
    两个LSTM的输出拼接后作为全连接层的输入，经过Dropout和BatchNormalization正则化，最终输出结果进行训练。
    :return:
    """
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    # Embedding层；将正整数（索引值）转换为固定尺寸的稠密向量。 例如： [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
    # 该层只能用作模型中的第一层。
    # 参数
    # input_dim: int > 0。词汇表大小， 即，最大整数 index + 1。
    # output_dim: int >= 0。词向量的维度。
    # embeddings_initializer: embeddings 矩阵的初始化方法 (详见 initializers)。
    # embeddings_regularizer: embeddings matrix 的正则化方法 (详见 regularizer)。
    # embeddings_constraint: embeddings matrix 的约束函数 (详见 constraints)。
    # mask_zero: 是否把 0 看作为一个应该被遮蔽的特殊的 "padding" 值。 这对于可变长的 循环神经网络层 十分有用。 如果设定为 True，那么接下来的所有层都必须支持 masking，否则就会抛出异常。 如果 mask_zero 为 True，作为结果，索引 0 就不能被用于词汇表中 （input_dim 应该与 vocabulary + 1 大小相同）。
    # input_length: 输入序列的长度，当它是固定的时。 如果你需要连接 Flatten 和 Dense 层，则这个参数是必须的 （没有它，dense 层的输出尺寸就无法计算）。
    # 输入尺寸
    # 尺寸为 (batch_size, sequence_length) 的 2D 张量。

    # 输出尺寸
    # 尺寸为 (batch_size, sequence_length, output_dim) 的 3D 张量。

    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    merged = concatenate([x1, y1])
    # concatenate, 连接一个输入张量的列表。
    # 它接受一个张量的列表， 除了连接轴之外，其他的尺寸都必须相同， 然后返回一个由所有输入张量连接起来的输出张量。
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)
    # BatchNormalization层在每个batch上将前一层的激活值重新规范化，即使得其输出数据的均值接近0，其标准差接近1

    merged = Dense(num_dense, activation=act)(merged)
    # Dense就是常用的全连接层

    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)
    preds = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[sequence_1_input, sequence_2_input],
                  outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['acc'])
    model.summary()
    return model
#
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# input_1 (InputLayer)            (None, 30)           0
# __________________________________________________________________________________________________
# input_2 (InputLayer)            (None, 30)           0
# __________________________________________________________________________________________________
# embedding_1 (Embedding)         (None, 30, 60)       2405820     input_1[0][0]
#                                                                  input_2[0][0]
# __________________________________________________________________________________________________
# lstm_1 (LSTM)                   (None, 175)          165200      embedding_1[0][0]
#                                                                  embedding_1[1][0]
# __________________________________________________________________________________________________
# concatenate_1 (Concatenate)     (None, 350)          0           lstm_1[0][0]
#                                                                  lstm_1[1][0]
# __________________________________________________________________________________________________
# dropout_1 (Dropout)             (None, 350)          0           concatenate_1[0][0]
# __________________________________________________________________________________________________
# batch_normalization_1 (BatchNor (None, 350)          1400        dropout_1[0][0]
# __________________________________________________________________________________________________
# dense_1 (Dense)                 (None, 100)          35100       batch_normalization_1[0][0]
# __________________________________________________________________________________________________
# dropout_2 (Dropout)             (None, 100)          0           dense_1[0][0]
# __________________________________________________________________________________________________
# batch_normalization_2 (BatchNor (None, 100)          400         dropout_2[0][0]
# __________________________________________________________________________________________________
# dense_2 (Dense)                 (None, 1)            101         batch_normalization_2[0][0]
# ==================================================================================================
# Total params: 2,608,021
# Trainable params: 201,301
# Non-trainable params: 2,406,720
# __________________________________________________________________________________________________

#######################################
# train the model
########################################


def train_model(nb_words, embedding_matrix, data_1, data_2, labels):
    """
    训练采用nAdam以及EarlyStopping，保存训练过程中验证集上效果最好的参数。最终对测试集进行预测。
    :return:
    """

    model = get_model(nb_words, embedding_matrix)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)

    hist = model.fit([data_1, data_2], labels,
                     validation_data=([data_1, data_2], labels),
                     epochs=2, batch_size=10, shuffle=True, callbacks=[early_stopping, model_checkpoint])

    # model.load_weights(bst_model_path)
    print('训练模型保存于：{}'.format(os.path.abspath(bst_model_path)))
    # model.save(bst_model_path)
    bst_score = min(hist.history['loss'])
    bst_acc = max(hist.history['acc'])
    print(bst_acc, bst_score)

def predict(question1, question2, bst_model_path='/home/gswyhq/github_projects/semanaly/model/lstm/lstm_175_100_0.15_0.15.h5',
            tokenizer_path="/home/gswyhq/github_projects/semanaly/model/lstm/tokenizer.pkl"):
    model = load_model(bst_model_path)
    tokenizer = pickle.load(open(os.path.join(tokenizer_path), 'rb'), encoding="iso-8859-1")
    texts_1 = [text_to_wordlist(question1)]
    texts_2 = [text_to_wordlist(question2)]
    sequences_1 = tokenizer.texts_to_sequences(texts_1)
    sequences_2 = tokenizer.texts_to_sequences(texts_2)
    data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
    data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
    p = model.predict([data_1, data_2])
    print(p)

if __name__ == '__main__':
    nb_words, embedding_matrix, data_1, data_2, labels, test_data_1, test_data_2, test_labels = generator_embedding_matrix(load_tokenizer=True)
    # print(labels)
    train_model(nb_words, embedding_matrix, data_1, data_2, labels)

# predicts = model.predict([test_data_1, test_data_2], batch_size=10, verbose=1)

# for i in range(len(test_ids)):
#    print "t1: %s, t2: %s, score: %s" % (test_texts_1[i], test_texts_2[i], predicts[i])
# 238710/238766 [============================>.] - ETA: 0s - loss: 0.5842 - acc: 0.6958
# 238730/238766 [============================>.] - ETA: 0s - loss: 0.5842 - acc: 0.6958
# 238750/238766 [============================>.] - ETA: 0s - loss: 0.5842 - acc: 0.6958
# 238766/238766 [==============================] - 856s 4ms/step - loss: 0.5842 - acc: 0.6958 - val_loss: 0.5585 - val_acc: 0.7129
# 0.6957523285795044 0.5842024635028623
