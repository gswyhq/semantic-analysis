#!/usr/bin/python3
# coding: utf-8
import codecs
import csv
import os
import pickle

import keras.preprocessing.text
import jieba
import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from config import save, EMBEDDING_FILE, TRAIN_DATA_FILE, TEST_DATA_FILE, tokenizer_name, \
    embedding_matrix_path, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, pretreatment_name

def read_data(train_data_path=TRAIN_DATA_FILE, val_data_path=TEST_DATA_FILE):
    texts_1 = []
    texts_2 = []
    labels = []
    with codecs.open(train_data_path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        # header = next(reader)
        for values in reader:
            texts_1.append(text_to_wordlist(values[0]))
            texts_2.append(text_to_wordlist(values[1]))
            labels.append(int(values[2].strip()))
    print('Found %s texts in train.csv' % len(texts_1))

    test_texts_1 = []
    test_texts_2 = []
    test_labels = []
    with codecs.open(val_data_path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        # header = next(reader)
        for values in reader:
            test_texts_1.append(text_to_wordlist(values[0]))
            test_texts_2.append(text_to_wordlist(values[1]))
            test_labels.append(int(values[2].strip()))
    print('Found %s texts in train.csv' % len(test_texts_1))
    return texts_1, texts_2, test_texts_1, test_texts_2, labels, test_labels

def text_to_wordlist(text, remove_stopwords=False):
    # Clean the text, with the option to remove stopwords.

    # Convert words to lower case and split them
    words = list(jieba.cut(text.strip(), cut_all=False))

    # Optionally, remove stop words
    if remove_stopwords:
        pass
    # Return a list of words
    return " ".join(words)


def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    if lower: text = text.lower()
    translate_table = {ord(c): ord(t) for c, t in zip(filters, split * len(filters))}

    text = text.translate(translate_table)
    seq = text.split(split)
    return [i for i in seq if i]

keras.preprocessing.text.text_to_word_sequence = text_to_word_sequence

'''
end here
语料编码
自然语言无法直接作为神经网络输入，需进行编码该部分包括以下步骤：

读人训练和测试数据，分词，并给每个词编号。
根据词编号，进一步生成每个句子的编号向量，句子采用固定长度，不足的位置补零。
保存词编号到文件，保存词向量矩阵方便预测使用。

中文分词使用jieba分词工具，词的编号则使用Keras的Tokenizer：

'''

def generator_embedding_matrix(load_tokenizer=False, train_data_path=TRAIN_DATA_FILE, val_data_path=TEST_DATA_FILE):


    if load_tokenizer:
        print('Load tokenizer...')
        tokenizer = pickle.load(open(tokenizer_name, 'rb'), encoding="iso-8859-1")
        pre_data = pickle.load(open(pretreatment_name, 'rb'), encoding='iso-8859-1')
        nb_words = pre_data.get("nb_words")
        embedding_matrix = pre_data.get("embedding_matrix")
        data_1 = pre_data.get("data_1")
        data_2 = pre_data.get("data_2")
        labels = pre_data.get("labels")
        test_data_1 = pre_data.get("test_data_1")
        test_data_2 = pre_data.get("test_data_2")
        test_labels = pre_data.get("test_labels")

    else:
        texts_1, texts_2, test_texts_1, test_texts_2, labels, test_labels = read_data(train_data_path, val_data_path)
        print("Fit tokenizer...")
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=False)
        tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)
        # 其中texts_1 、texts_2 、test_texts_1 、 test_texts_2的元素分别为训练数据和测试数据的分词后的列表
        if save:
            print("Save tokenizer...")
            if not os.path.exists(os.path.split(tokenizer_name)[0]):
                os.makedirs(os.path.split(tokenizer_name)[0])
            pickle.dump(tokenizer, open(tokenizer_name, "wb"), protocol=2)

        # 利用tokenizer对语料中的句子进行编号
        # > sequences_1 = tokenizer.texts_to_sequences(texts_1)
        # > print sequences_1
        # [[2 1 3], ...]

        sequences_1 = tokenizer.texts_to_sequences(texts_1)
        sequences_2 = tokenizer.texts_to_sequences(texts_2)
        test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
        test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)
        # print sequences_1

        # 经过上面的过程 tokenizer保存了语料中出现过的词的编号映射。
        # > print tokenizer.word_index
        # {"我"： 2， "是"：1， "谁"：3}

        word_index = tokenizer.word_index
        print('Found %s unique tokens' % len(word_index))

        # 最终生成固定长度(假设为10)的句子编号列表
        # > data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
        # > print data_1
        # [[0 0 0 0 0 0 0 2 1 3], ...]
        # data_1即可作为神经网络的输入。

        data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
        data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
        labels = np.array(labels)
        print('Shape of data tensor:', data_1.shape)
        print('Shape of label tensor:', labels.shape)

        test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
        test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
        test_labels = np.array(test_labels)

        ########################################
        # prepare embeddings
        # 词向量映射
        # 在对句子进行编码后，需要准备句子中词的词向量映射作为LSTM层的输入。这里使用预训练的词向量（这里）参数，生成词向量映射矩阵：
        ########################################

        print('Preparing embedding matrix')
        word2vec = Word2Vec.load(EMBEDDING_FILE)

        nb_words = min(MAX_NB_WORDS, len(word_index)) + 1

        unk_words_num = 0
        embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
        for word, i in word_index.items():
            if word in word2vec.wv.vocab:
                embedding_matrix[i] = word2vec.wv.word_vec(word)
            else:
                print (word)
                unk_words_num += 1
        print('总共有{}个词不存在词向量'.format(unk_words_num))
        print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

        np.save(embedding_matrix_path, embedding_matrix)

        data = {'nb_words': nb_words,
                 'embedding_matrix': embedding_matrix,
                 'data_1': data_1,
                 'data_2': data_2,
                 'labels': labels,
                 'test_data_1': test_data_1,
                 'test_data_2': test_data_2,
                 'test_labels': test_labels
                }

        pickle.dump(data, open(pretreatment_name, "wb"), protocol=2)

    return nb_words, embedding_matrix, data_1, data_2, labels, test_data_1, test_data_2, test_labels

def main():
    nb_words, embedding_matrix, data_1, data_2, labels, test_data_1, test_data_2, test_labels = generator_embedding_matrix()


if __name__ == '__main__':
    main()

