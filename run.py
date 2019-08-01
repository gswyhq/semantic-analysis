import numpy as np
import spacy
import tensorflow as tf
import sys
import os
import argparse
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.sequence import pad_sequences

from word2vector.embedding_matrix import generator_embedding_matrix
from esim.esim import get_model as esim_model
from lstm.lstm import get_model as lstm_model
from utils import to_categorical
from config import MAX_SEQUENCE_LENGTH, SAVE_MODEL_PATH, SAVE_LOG_PATH

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = None


def do_pred(test_data_path):
    if FLAGS.load_model is None:
        raise ValueError("You need to specify the model location by --load_model=[location]")

    # # Load Testing Data
    # question_1, question_2 = get_test_from_csv(test_data_path)
    #
    # # Load Pre-trained Model
    # if FLAGS.best_glove:
    #     import en_core_web_md
    #     nlp = en_core_web_md.load()  # load best-matching version for Glove
    # else:
    #     nlp = spacy.load('en')
    # embedding_matrix = load_glove_embeddings(nlp.vocab, n_unknown=FLAGS.num_unknown)  # shape=(1071074, 300)
    #
    # tf.logging.info('Build model ...')
    # esim = ESIM(embedding_matrix, FLAGS.max_length, FLAGS.num_hidden, FLAGS.num_classes, FLAGS.keep_prob, FLAGS.learning_rate)
    #
    # if FLAGS.load_model:
    #     model = esim.build_model(FLAGS.load_model)
    # else:
    #     raise ValueError("You need to specify the model location by --load_model=[location]")
    #
    # # Convert the "raw data" to word-ids format && convert "labels" to one-hot vectors
    # q1_test, q2_test = convert_questions_to_word_ids(question_1, question_2, nlp, max_length=FLAGS.max_length, tree_truncate=FLAGS.tree_truncate)
    #
    # predictions = model.predict([q1_test, q2_test])
    # print("[*] Predictions Results: \n", predictions[0])

def do_eval(test_data_path, shuffle=False):
    if FLAGS.load_model is None:
        raise ValueError("You need to specify the model location by --load_model=[location]")

    # # Load Testing Data
    # question_1, question_2, labels = get_input_from_csv(test_data_path)
    #
    # if shuffle:
    #     question_1, question_2, labels = shuffle_data(question_1, question_2, labels)
    #
    # # Load Pre-trained Model
    # if FLAGS.best_glove:
    #     import en_core_web_md
    #     nlp = en_core_web_md.load()  # load best-matching version for Glove
    # else:
    #     nlp = spacy.load('en')
    # embedding_matrix = load_glove_embeddings(nlp.vocab, n_unknown=FLAGS.num_unknown)  # shape=(1071074, 300)
    #
    # tf.logging.info('Build model ...')
    # esim = ESIM(embedding_matrix, FLAGS.max_length, FLAGS.num_hidden, FLAGS.num_classes, FLAGS.keep_prob, FLAGS.learning_rate)
    #
    # if FLAGS.load_model:
    #     model = esim.build_model(FLAGS.load_model)
    # else:
    #     raise ValueError("You need to specify the model location by --load_model=[location]")
    #
    # # Convert the "raw data" to word-ids format && convert "labels" to one-hot vectors
    # q1_test, q2_test = convert_questions_to_word_ids(question_1, question_2, nlp, max_length=FLAGS.max_length, tree_truncate=FLAGS.tree_truncate)
    # labels = to_categorical(np.asarray(labels, dtype='int32'))
    #
    # scores = model.evaluate([q1_test, q2_test], labels, batch_size=FLAGS.batch_size, verbose=1)
    #
    # print("=================== RESULTS =====================")
    # print("[*] LOSS OF TEST DATA: %.4f" % scores[0])
    # print("[*] ACCURACY OF TEST DATA: %.4f" % scores[1])


def train(train_data_path, val_data_path, batch_size, n_epochs, save_dir=None):
    # Stage 1: Read training data (csv) && Preprocessing them
    tf.logging.info('Loading training and validataion data ...')
    nb_words, embedding_matrix, q1_train, q2_train, train_labels, test_data_1, test_data_2, test_labels = generator_embedding_matrix(train_data_path=train_data_path, val_data_path=val_data_path, load_tokenizer=True)

    # Stage 2: Load Pre-trained embedding matrix (Using GLOVE here)
    tf.logging.info('Loading pre-trained embedding matrix ...')

    # Stage 3: Build Model
    tf.logging.info('Build model ...')

    if FLAGS.model == 'lstm':
        model = lstm_model(nb_words, embedding_matrix)
    elif FLAGS.model == 'esim':
        model = esim_model(embedding_matrix, MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)
        train_labels = to_categorical(np.asarray(train_labels, dtype='int32'))
        test_labels = to_categorical(np.asarray(test_labels, dtype='int32'))
    else:
        raise ValueError('该方法暂无实现：{}'.format(FLAGS.model))

    # Stage 4: Convert the "raw data" to word-ids format && convert "labels" to one-hot vectors
    tf.logging.info('Converting questions into ids ...')
    # q1_train = pad_sequences(train_question_1, maxlen=FLAGS.max_length)
    # q2_train = pad_sequences(train_question_2, maxlen=FLAGS.max_length)


    # q1_train, q2_train = convert_questions_to_word_ids(train_question_1, train_question_2, nlp, max_length=FLAGS.max_length, tree_truncate=FLAGS.tree_truncate)


    # q1_val, q2_val = convert_questions_to_word_ids(val_question_1, val_question_2, nlp, max_length=FLAGS.max_length, tree_truncate=FLAGS.tree_truncate)
    # val_labels = to_categorical(np.asarray(val_labels, dtype='int32'))

    # Stage 5: Training
    tf.logging.info('Start training ...')

    callbacks = []
    save_dir = save_dir if save_dir is not None else 'checkpoints'
    filepath = os.path.join(save_dir, FLAGS.model, "model-{epoch:02d}-{val_acc:.2f}.hdf5")
    if not os.path.isdir(os.path.join(save_dir, FLAGS.model)):
        os.makedirs(os.path.join(save_dir, FLAGS.model))

    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', save_weights_only=False)
    callbacks.append(checkpoint)

    tf.logging.info('tensorboard')
    if FLAGS.tensorboard:
        graph_dir = os.path.join('.', 'GRAPHs')
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)
        tb = TensorBoard(log_dir=graph_dir, histogram_freq=0, write_graph=True, write_images=True)
        callbacks.append(tb)
    csv_logger = CSVLogger(os.path.join(SAVE_LOG_PATH, FLAGS.model, 'training.log'), append=True, separator=';')
    if not os.path.isdir(os.path.join(SAVE_LOG_PATH, FLAGS.model)):
        os.makedirs(os.path.join(SAVE_LOG_PATH, FLAGS.model))
    callbacks.append(csv_logger)
    tf.logging.info('开始训练...')
    history_callback = model.fit(
        x=[q1_train, q2_train],
        y=train_labels,
        batch_size=batch_size,
        epochs=n_epochs,
        validation_data=([test_data_1, test_data_2], test_labels),
        # validation_split=0.2,
        callbacks=callbacks,
        shuffle=True,
        verbose=FLAGS.verbose
    )

    loss_history = history_callback.history.get('loss')

    tf.logging.info('训练完成！')

def run(_):
    if FLAGS.mode == 'train':
        train(FLAGS.input_data, FLAGS.val_data, FLAGS.batch_size, FLAGS.num_epochs, save_dir=SAVE_MODEL_PATH)
    elif FLAGS.mode == 'eval':
        do_eval(FLAGS.input_data)
    elif FLAGS.mode == 'pred':
        do_pred(FLAGS.test_data)
    else:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=2,
        help='Specify number of epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Specify number of batch size'
    )
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=300,
        help='Specify embedding size'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=30,
        help='Specify the max length of input sentence'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=10,
        help='Specify seed for randomization'
    )
    parser.add_argument(
        '--input_data',
        type=str,
        default="/home/gswyhq/data/LCQMC/train.txt",
        help='Specify the location of input data',
    )
    parser.add_argument(
        '--test_data',
        type=str,
        default="/home/gswyhq/data/LCQMC/test.txt",
        help='Specify the location of test data',
    )
    parser.add_argument(
        '--val_data',
        type=str,
        default="/home/gswyhq/data/LCQMC/dev.txt",
        help='Specify the location of test data',
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=2,
        help='Specify the number of classes'
    )
    parser.add_argument(
        '--num_hidden',
        type=int,
        default=100,
        help='Specify the number of hidden units in each rnn cell'
    )
    parser.add_argument(
        '--num_unknown',
        type=int,
        default=100,
        help='Specify the number of unknown words for putting in the embedding matrix'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=4e-4,
        help='Specify dropout rate'
    )
    parser.add_argument(
        '--keep_prob',
        type=float,
        default=0.8,
        help='Specify the rate (between 0 and 1) of the units that will keep during training'
    )
    parser.add_argument(
        '--best_glove',
        action='store_true',
        help='Glove: using light version or best-matching version',
    )
    parser.add_argument(
        '--tree_truncate',
        action='store_true',
        help='Specify whether do tree_truncate or not',
        default=False
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose on training',
        default=False
    )
    parser.add_argument(
        '--load_model',
        type=str,
        help='Locate the path of the model',
    )
    parser.add_argument(
        '--tensorboard',
        action='store_true',
        help='Whether use tensorboard or not',
        default=True
    )
    parser.add_argument(
        '--mode',
        type=str,
        help='Specify mode: train or eval',
        required=True
    )
    parser.add_argument(
        '--model',
        type=str,
        help='模型类型',
        default='lstm'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=run, argv=[sys.argv[0]] + unparsed)
