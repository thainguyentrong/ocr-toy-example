# pylint: disable=missing-module-docstring, missing-function-docstring, line-too-long, invalid-name
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from hyperparams import Hyperparams as hp
from model import Model
from dataloader import input_fn, load_vocab


def predict(testset, max_sequence_length):
    img_w, img_h = hp.size
    images_input = tf.placeholder(shape=(1, img_h, img_w, 1), dtype=tf.float32)
    sequences_input = tf.placeholder(shape=(1, max_sequence_length), dtype=tf.int32)
    is_training = tf.placeholder(shape=(), dtype=tf.bool)


    model = Model(images_input, sequences_input, is_training, max_sequence_length)
    endpoints = model.endpoints()
    saver = tf.train.Saver()

    test_dataset = input_fn(testset, batch_size=1, max_sequence_length=max_sequence_length)
    test_iterator = test_dataset.make_one_shot_iterator()
    test_node = test_iterator.get_next()

    _, idx2char = load_vocab()

    with tf.Session() as sess:
        latest = tf.train.latest_checkpoint(hp.logdir)
        try:
            saver.restore(sess, latest)
            print('Restore model')
        except:
            print("Restore error")

        for _ in range(len(testset)):
            img, _, _ = sess.run(test_node)
            img = np.expand_dims(img, axis=-1)
            predictions, alignments = sess.run([endpoints['predictions'], endpoints['alignments']], feed_dict={images_input: img, is_training: False})
            predicted_text = ''.join(idx2char[x] for x in predictions[0] if x < len(hp.alphabet)).strip()
            print("Predicted: {}".format(predicted_text))
            img = np.squeeze(img, axis=0)
            img = np.squeeze(img, axis=-1)

            res_img = []
            for ind, alignment in enumerate(alignments[0]):
                if ind == len(predicted_text):
                    break
                h, w = img.shape[:2]
                img_al = cv2.resize(alignment, (w, h), interpolation=cv2.INTER_AREA)
                highlighted = cv2.resize((img + img_al*4e+2) / 4e+2, (500, 500), interpolation=cv2.INTER_AREA)
                res = Image.fromarray(np.array(highlighted * 255/ np.amax(highlighted)).astype(np.uint8), 'L')
                draw = ImageDraw.Draw(res)
                font = ImageFont.truetype('./font/arial.ttf', 50)
                draw.text((50, 50), predicted_text[ind], (255), font=font)

                res_img.append(res)
                np_img = np.array(res)
                cv2.imshow('alignment', np_img)
                k = cv2.waitKey(1000) & 0xFF
                if k == 27:
                    break

            res_img[0].save('result.gif', save_all=True, append_images=res_img[1:], duration=50 * (ind + 1), loop=10000)
            k = cv2.waitKey(0) & 0xFF
            if k == 27:
                break


if __name__ == '__main__':
    with open('./test.txt', 'r', encoding='utf-8') as f:
        testset = f.readlines()
    testset = [text.strip() for text in testset]
    predict(testset, max_sequence_length=32)
