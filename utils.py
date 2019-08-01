import numpy as np
import re
import os
from gensim.models import Word2Vec


def to_categorical(y, nb_classes=None):
    y = np.asarray(y, dtype='int32')

    if not nb_classes:
        nb_classes = np.max(y) + 1

    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.

    return Y


def shuffle_data(question_1, question_2, labels):
    q1 = []
    q2 = []
    y = []

    shuffle_indices = np.random.permutation(np.arange(len(question_1)))
    for index in shuffle_indices:
        q1.append(question_1[index])
        q2.append(question_2[index])
        y.append(labels[index])

    return q1, q2, y
