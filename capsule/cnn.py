

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa

from capsule.reconstruction_network import ReconstructionNetwork


class CNN(tf.keras.Model):
    

    def __init__(self, args):
        super(CNN, self).__init__()

        self.dimensions = list(map(int, args.dimensions.split(","))) if args.dimensions != "" else []
        self.layers = list(map(int, args.layers.split(","))) if args.layers != "" else []
        self.use_bias=args.use_bias
        self.use_reconstruction=args.use_reconstruction
        self.num_classes = layers[-1]
        self.args = args
        self.fcs = []
        self.convs = []
        

        self.convs.append(tf.keras.layers.Conv2D(filters=8,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same",
                                            activation="relu"))

        self.fc1 = layers.Dense(512, 
            name="fc-1", 
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(l2),
            kernel_initializer="he_normal")
        
        self.out = layers.Dense(2, 
            name="out", 
            activation="linear",
            kernel_regularizer=tf.keras.regularizers.l2(l2))
        
        if self.use_reconstruction:
            self.reconstruction_network = ReconstructionNetwork(
                name="ReconstructionNetwork",
                in_capsules=self.num_classes, 
                in_dim=dimensions[-1],
                out_dim=args.img_height,
                img_dim=args.img_depth)


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

        r = self.reconstruction_network(x, y) if self.use_reconstruction else None
        out = self.out(x)
        return out, r, layers
