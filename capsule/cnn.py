

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa

from capsule.reconstruction_network import ReconstructionNetwork


class CNN(tf.keras.Model):
    

    def __init__(self, args):
        super(CNN, self).__init__()

        dimensions = list(map(int, args.dimensions.split(","))) if args.dimensions != "" else []
        layers = list(map(int, args.layers.split(","))) if args.layers != "" else []
        self.use_bias=args.use_bias
        self.use_reconstruction=args.use_reconstruction
        self.num_classes = layers[-1]
        self.args = args
        self.fcs = []
        self.convs = []
        channels = layers[0]
        dim = dimensions[0]

        self.convs.append(tf.keras.layers.Conv2D(
            filters=channels * dim,
            kernel_size=(9, 9),
            strides=2,
            padding="same",
            activation="relu"))

        for i in range(1, len(layers)):
            self.fcs.append(tf.keras.layers.Dense(
                dimensions[i] * layers[i], 
                activation="relu"))
        
        self.out = tf.keras.layers.Dense(self.num_classes, 
            name="out", 
            activation="linear",
            use_bias=self.use_bias)


    def call(self, x, y):
        batch_size = tf.shape(x)[0]
        layers = []

        for conv in self.convs:
            x = conv(x)
            layers.append(x)

        x = tf.reshape(x, [batch_size, -1])
        for fc in self.fcs:
            x = fc(x)
            layers.append(x)

        out = self.out(x)
        return out, None, layers
