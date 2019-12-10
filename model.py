# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring, line-too-long, invalid-name
import tensorflow as tf
from hyperparams import Hyperparams as hp
from vgg import VGG_Network

class Model:
    def __init__(self, images_input, sequences_input, is_training, max_sequence_length):
        self.max_sequence_length = max_sequence_length
        self.alphabet_size = len(hp.alphabet)
        self.batch_size = images_input.shape.as_list()[0]

        features = self.conv_layers(images_input, is_training)
        features_h = tf.shape(features)[1]
        features = self.flatten(features)

        sequences, start_tokens, lengths, weights = self.prepare_sequences(seq=sequences_input)
        logits, self.alignments, self.predictions = self.create_attention(memory=features, sequences=sequences, lengths=lengths, features_h=features_h, start_tokens=start_tokens)
        self.loss = self.create_loss(logits, sequences_input, weights)

    def endpoints(self):
        return {
            "loss": self.loss,
            "alignments": self.alignments,
            "predictions": self.predictions,
        }

    def conv_layers(self, inputs, is_training):
        module = VGG_Network(inputs, is_training)
        features = module.get_features()
        # 1x1 convolution for dimention reduction
        features = tf.layers.conv2d(inputs=features, filters=64, kernel_size=(1, 1), padding="same", activation=tf.nn.relu)
        return features

    def flatten(self, features):
        feature_size = features.get_shape().dims[3].value
        return tf.reshape(features, [self.batch_size, -1, feature_size])

    def add_start_tokens(self, seq, start_sym):
        start_tokens = tf.ones([self.batch_size], dtype=tf.int32)*start_sym
        return tf.concat([tf.expand_dims(start_tokens, 1), seq], axis=1), start_tokens

    def prepare_sequences(self, seq):
        lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(seq, self.alphabet_size)), axis=1)
        lengths = lengths + 1

        weights = tf.cast(tf.sequence_mask(lengths, self.max_sequence_length), dtype=tf.float32)
        seq_train, start_tokens = self.add_start_tokens(seq=seq, start_sym=self.alphabet_size + 1)
        sequences = tf.contrib.layers.one_hot_encoding(seq_train, num_classes=self.alphabet_size + 2)
        return sequences, start_tokens, lengths, weights

    def create_attention(self, memory, sequences, lengths, features_h, start_tokens, num_units=hp.num_units):
        train_helper = tf.contrib.seq2seq.TrainingHelper(inputs=sequences, sequence_length=lengths)
        embeddings = lambda x: tf.contrib.layers.one_hot_encoding(x, self.alphabet_size + 2)
        pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, start_tokens=start_tokens, end_token=self.alphabet_size)

        def decode(helper, reuse=False):
            with tf.variable_scope('decode', reuse=reuse):
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units, memory)
                cell = tf.nn.rnn_cell.GRUCell(num_units)
                attention_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, alignment_history=True, name="attention")
                output_cell = tf.contrib.rnn.OutputProjectionWrapper(attention_cell, self.alphabet_size + 2, reuse=reuse)
                decoder = tf.contrib.seq2seq.BasicDecoder(cell=output_cell, helper=helper, initial_state=output_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size))
                outputs = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=False, impute_finished=True, maximum_iterations=self.max_sequence_length)
            return outputs

        def alignments_full(alignments):
            depth = tf.shape(alignments)[0]
            alignments = tf.transpose(alignments, [1, 0, 2])
            alignments = tf.reshape(alignments, [self.batch_size, depth, features_h, -1])
            return alignments

        def complete(logits, to_complete):
            zeros_first = tf.zeros((self.batch_size, to_complete, self.alphabet_size), dtype=tf.float32)
            zeros_last = tf.zeros((self.batch_size, to_complete, 1), dtype=tf.float32)
            ones = tf.ones((self.batch_size, to_complete, 1), dtype=tf.float32)
            completion = tf.concat([zeros_first, ones, zeros_last], axis=2)
            return tf.concat([logits, completion], axis=1)

        train_outputs = decode(train_helper)
        pred_outputs = decode(pred_helper, reuse=True)
        alignments = alignments_full(pred_outputs[1].alignment_history.stack())
        logits = train_outputs[0].rnn_output
        to_complete = self.max_sequence_length - tf.shape(logits)[1]
        logits = tf.cond(tf.equal(to_complete, 0), lambda: logits, lambda: complete(logits, to_complete))
        return logits, alignments, tf.argmax(tf.nn.sigmoid(pred_outputs[0].rnn_output), axis=2)

    def create_loss(self, logits, targets, weights):
        return tf.contrib.seq2seq.sequence_loss(logits=logits, targets=targets, weights=weights)
    