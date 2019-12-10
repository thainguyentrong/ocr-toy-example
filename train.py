# pylint: disable=missing-module-docstring, missing-function-docstring, line-too-long, invalid-name
import sys
import time
import tqdm
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from underthesea import word_tokenize
from hyperparams import Hyperparams as hp
from model import Model
from dataloader import input_fn, load_vocab


def train(trainset, validset, max_sequence_length):
    img_w, img_h = hp.size
    images_input = tf.placeholder(shape=(hp.batch_size, img_h, img_w, 1), dtype=tf.float32)
    sequences_input = tf.placeholder(shape=(hp.batch_size, max_sequence_length), dtype=tf.int32)
    is_training = tf.placeholder(shape=(), dtype=tf.bool)

    model = Model(images_input, sequences_input, is_training, max_sequence_length)
    endpoints = model.endpoints()

    global_step = tf.train.get_or_create_global_step()
    train_op = tf.contrib.layers.optimize_loss(endpoints['loss'], global_step, optimizer='Adam', learning_rate=1e-3, summaries=['loss', 'learning_rate'])
    saver = tf.train.Saver()

    train_dataset = input_fn(trainset, batch_size=hp.batch_size, max_sequence_length=max_sequence_length)
    train_iterator = train_dataset.make_one_shot_iterator()
    train_node = train_iterator.get_next()

    valid_dataset = input_fn(validset, batch_size=hp.batch_size, max_sequence_length=max_sequence_length)
    valid_iterator = valid_dataset.make_one_shot_iterator()
    valid_node = valid_iterator.get_next()

    trainable = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print('Trainable parameters: ', trainable)
    _, idx2char = load_vocab()

    with tf.Session() as sess:
        latest = tf.train.latest_checkpoint(hp.logdir)
        if latest:
            saver.restore(sess, latest)
            print('Restore model')
        else:
            sess.run(tf.global_variables_initializer())

        for _ in range(hp.steps):
            start = time.time()
            imgs, seqs, _ = sess.run(train_node)
            imgs = np.expand_dims(imgs, axis=-1)

            _, loss, gs = sess.run([train_op, endpoints['loss'], global_step], feed_dict={images_input: imgs, sequences_input: seqs, is_training: True})
            end = time.time()
            sys.stdout.write('\rStep: %06d, Loss: %.5f, Training Time: %.2f secs' % (gs, loss, end - start))

            if gs % hp.display == 0:
                vsteps = len(validset) // hp.batch_size
                pred_text, gt_texts = [], []
                start = time.time()
                for _ in range(vsteps):
                    igs, sqs, texts = sess.run(valid_node)
                    igs = np.expand_dims(igs, axis=-1)
                    gt_texts += [t.decode() for t in texts]
                    predictions = sess.run(endpoints['predictions'], feed_dict={images_input: igs, sequences_input: sqs, is_training: False})
                    pred_text += [''.join(idx2char[x] for x in p if x < len(hp.alphabet)).strip() for p in predictions]

                end = time.time()
                print("\r\nValidation Time: %.2f secs" % (end - start))
                print("Ground truth:\t", sqs[0])
                print("Prediction:\t", predictions[0])
                print("Accuracy: %.2f" % (np.sum(np.array(pred_text) == np.array(gt_texts))/len(validset)))
                saver.save(sess, hp.logdir + 'model')

if __name__ == '__main__':
    with open('./data.txt', 'r', encoding='utf-8') as f:
        data = f.readlines()
    lines = []
    for text in tqdm.tqdm(data):
        lines += word_tokenize(text.strip())
    max_sequence_length = np.amax([len(text.strip()) for text in lines]) + 1 # end sym
    print('Max sequence length: ', max_sequence_length)

    trainset, validset = train_test_split(lines, test_size=0.1, random_state=42)
    train(trainset, validset, max_sequence_length)
