#!/usr/bin/python3
# coding: utf-8


import os
import sys
import pickle

import numpy as np
import tensorflow as tf
from keras.layers import Lambda, Activation, Dropout, Embedding, SpatialDropout1D, Dense, merge
from keras.optimizers import Adam
from keras.layers import TimeDistributed  # This applies the model to every timestep in the input sequences
from keras.layers import Bidirectional, LSTM
# from keras.layers.advanced_activations import ELU
from keras.models import Sequential
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Bidirectional, Lambda, K, Softmax, Dot, Multiply, \
    Concatenate, Subtract
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from word2vector.embedding_matrix import generator_embedding_matrix

def get_model(embedding_matrix, MAX_SEQUENCE_LENGTH=30):
    """
    网络结构
    该神经网络采用简单的单层LSTM+全连接层对数据进行训练，网络结构图
    该部分首先定义embedding_layer作为输入层和LSTM层的映射层，将输入的句子编码映射为词向量列表作为LSTM层的输入。
    两个LSTM的输出拼接后作为全连接层的输入，经过Dropout和BatchNormalization正则化，最终输出结果进行训练。
    :return:
    """
    nb_words = embedding_matrix.shape[0]
    EMBEDDING_DIM = embedding_matrix.shape[1]

    a = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')
    b = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')

    x1 = Embedding(nb_words,
              EMBEDDING_DIM,
              weights=[embedding_matrix],
              input_length=MAX_SEQUENCE_LENGTH,
              trainable=False)(a)
    x2 = Embedding(nb_words,
              EMBEDDING_DIM,
              weights=[embedding_matrix],
              input_length=MAX_SEQUENCE_LENGTH,
              trainable=False)(b)

    x1 = Bidirectional(LSTM(100, return_sequences=True))(x1)
    x2 = Bidirectional(LSTM(100, return_sequences=True))(x2)

    e = Dot(axes=2)([x1, x2])
    e1 = Softmax(axis=2)(e)
    e2 = Softmax(axis=1)(e)
    e1 = Lambda(K.expand_dims, arguments={'axis': 3})(e1)
    e2 = Lambda(K.expand_dims, arguments={'axis': 3})(e2)

    _x1 = Lambda(K.expand_dims, arguments={'axis': 1})(x2)
    _x1 = Multiply()([e1, _x1])
    _x1 = Lambda(K.sum, arguments={'axis': 2})(_x1)
    _x2 = Lambda(K.expand_dims, arguments={'axis': 2})(x1)
    _x2 = Multiply()([e2, _x2])
    _x2 = Lambda(K.sum, arguments={'axis': 1})(_x2)

    m1 = Concatenate()([x1, _x1, Subtract()([x1, _x1]), Multiply()([x1, _x1])])
    m2 = Concatenate()([x2, _x2, Subtract()([x2, _x2]), Multiply()([x2, _x2])])

    y1 = Bidirectional(LSTM(100, return_sequences=True))(m1)
    y2 = Bidirectional(LSTM(100, return_sequences=True))(m2)

    mx1 = Lambda(K.max, arguments={'axis': 1})(y1)
    av1 = Lambda(K.mean, arguments={'axis': 1})(y1)
    mx2 = Lambda(K.max, arguments={'axis': 1})(y2)
    av2 = Lambda(K.mean, arguments={'axis': 1})(y2)

    y = Concatenate()([av1, mx1, av2, mx2])
    y = Dense(120, activation='tanh')(y)
    y = Dropout(0.5)(y)
    y = Dense(120, activation='tanh')(y)
    y = Dropout(0.5)(y)
    y = Dense(2, activation='softmax')(y)

    model = Model(inputs=[a, b], outputs=y)
    model.compile(optimizer=Adam(lr=4e-4), loss='binary_crossentropy', metrics=['accuracy'])

    print(model.summary())

    return model
#


def main():
    nb_words, embedding_matrix, data_1, data_2, labels, test_data_1, test_data_2, test_labels = generator_embedding_matrix(load_tokenizer=True)
    model = get_model(embedding_matrix, MAX_SEQUENCE_LENGTH=30)


if __name__ == '__main__':
    main()