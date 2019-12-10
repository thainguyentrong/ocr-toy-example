# pylint: disable=line-too-long, missing-module-docstring, missing-function-docstring, missing-final-newline
import textwrap
import functools
import random
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from hyperparams import Hyperparams as hp


def load_vocab():
    char2idx = {char: idx for idx, char in enumerate(hp.alphabet)}
    idx2char = {idx: char for idx, char in enumerate(hp.alphabet)}
    return char2idx, idx2char

def create_data(text, max_sequence_length):
    font_name = np.random.choice(a=['arial.ttf', 'amazone.ttf', 'cour.ttf', 'tahoma.ttf', 'timesi.ttf', 'palab.ttf'])
    img = np.zeros((hp.size[1], hp.size[0]), dtype=np.uint8)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('./font/' + font_name, 30)

    lines = textwrap.wrap(text, width=9)
    y_text = (hp.size[1] - font.getsize(lines[0])[1]*len(lines)) / 2
    for line in lines:
        width, height = font.getsize(line)
        draw.text(((hp.size[0] - width) / 2, y_text), line, fill=(255), font=font)
        y_text += height

    char2idx, _ = load_vocab()
    completion = (max_sequence_length - len(text))
    seq = [char2idx[c] for c in list(text)] + [len(hp.alphabet)] * completion # end token

    return np.array(img), np.array(seq, dtype=np.int32)

def generator_fn(lines, batch_size, max_sequence_length):
    imgs, seqs, texts = [], [], []
    while True:
        random.shuffle(lines)
        for text in lines:
            img, seq = create_data(text.strip(), max_sequence_length)
            imgs.append(img)
            seqs.append(seq)
            texts.append(text.strip().encode())
            if len(imgs) == batch_size:
                yield imgs, seqs, texts
                imgs, seqs, texts = [], [], []

def input_fn(lines, batch_size, max_sequence_length):
    shapes = ((None, hp.size[1], hp.size[0]), (None, None), (None))
    types = (tf.float32, tf.int32, tf.string)
    dataset = tf.data.Dataset.from_generator(functools.partial(generator_fn, lines, batch_size, max_sequence_length), output_shapes=shapes, output_types=types)
    return dataset
