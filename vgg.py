# pylint: disable=missing-module-docstring, missing-class-docstring, line-too-long, missing-function-docstring, invalid-name
import tensorflow as tf

class VGG_Network:
    def __init__(self, inputs, is_training):
        self.features = self.network(inputs, is_training)

    def get_features(self):
        return self.features

    def conv_block(self, num_layer, inputs, filters, is_training, name):
        with tf.variable_scope(name):
            conv = inputs
            for i in range(num_layer):
                conv = tf.layers.conv2d(inputs=conv, filters=filters, kernel_size=(3, 3), padding="same", activation=tf.nn.relu, name='conv_%d' % i)
            norm = tf.layers.batch_normalization(inputs=conv, training=is_training, scale=False, momentum=0.99)
            return norm

    def network(self, inputs, is_training):
        features = self.conv_block(2, inputs, 32, is_training, 'conv_block_1')
        pool = tf.layers.max_pooling2d(inputs=features, pool_size=(2, 2), strides=2)

        features = self.conv_block(2, pool, 64, is_training, 'conv_block_2')
        pool = tf.layers.max_pooling2d(inputs=features, pool_size=(2, 2), strides=2)

        features = self.conv_block(3, pool, 128, is_training, 'conv_block_3')
        pool = tf.layers.max_pooling2d(inputs=features, pool_size=(2, 2), strides=2)

        features = self.conv_block(3, pool, 256, is_training, 'conv_block_4')
        features = self.conv_block(3, features, 256, is_training, 'conv_block_5')

        return features
    