#!/usr/bin/python3
# coding: utf-8
import os

save = True

# EMBEDDING_FILE = '../model/w2v/w2v.mod'
# https://chiang97912.github.io/2018/01/08/开源啦！60维维基百科词向量免费放送/
# https://pan.baidu.com/s/1o8f1ELs
EMBEDDING_FILE = '/home/gswyhq/data/WordVector_60dimensional/wiki.zh.text.model'
TRAIN_DATA_FILE = '/home/gswyhq/data/LCQMC/train.txt'
TEST_DATA_FILE = '/home/gswyhq/data/LCQMC/test.txt'

# EMBEDDING_FILE = '/notebooks/data/WordVector_60dimensional/wiki.zh.text.model'
# TRAIN_DATA_FILE = '/notebooks/data/LCQMC/train.txt'
# TEST_DATA_FILE = '/notebooks/data/LCQMC/test.txt'


tokenizer_name = "./model/tokenizer.pkl"
pretreatment_name = "./model/pretreatment.pkl"
embedding_matrix_path = "./model/embedding_matrix.npy"
lstm_bst_model_path = './model/lstm/lstm_175_100_0.15_0.15.h5'
esim_bst_model_path = './mode/esim/model.h5'



# 保存模型的路径

MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 800000
EMBEDDING_DIM = 60  # 向量的维数
VALIDATION_SPLIT = 0.1

num_lstm = 175
num_dense = 100
rate_drop_lstm = 0.15
rate_drop_dense = 0.15

SAVE_MODEL_PATH = './model'
SAVE_LOG_PATH = './log'
act = 'relu'
re_weight = True  # whether to re-weight classes to fit the 17.5% share in test set


def main():
    pass


if __name__ == '__main__':
    main()
