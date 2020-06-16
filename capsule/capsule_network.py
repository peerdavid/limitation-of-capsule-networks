

import tensorflow as tf

from capsule.capsule_layer import Capsule
from capsule.em_capsule_layer import EMCapsule
from capsule.primary_capsule_layer import PrimaryCapsule
from capsule.reconstruction_network import ReconstructionNetwork
from capsule.norm_layer import Norm


class CapsNet(tf.keras.Model):

    def __init__(self, routing, layers, dimensions, use_bias=False, use_reconstruction=True):
        super(CapsNet, self).__init__()

        CapsuleType = {
            "rba": Capsule,
            "em": EMCapsule
        }

        self.use_bias=use_bias
        self.use_reconstruction = use_reconstruction
        self.num_classes = layers[-1]

        with tf.name_scope(self.name):
            self.reshape = tf.keras.layers.Reshape(target_shape=[28, 28, 1], input_shape=(28, 28,))

            channels = layers[0]
            dim = dimensions[0]
            self.conv_1 = tf.keras.layers.Conv2D(channels * dim, (9, 9), padding='valid', activation=tf.nn.relu)
            self.primary = PrimaryCapsule(name="PrimaryCapsuleLayer", channels=channels, dim=dim, kernel_size=(9, 9))
            self.capsule_layers = []

            for i in range(1, len(layers)):
                self.capsule_layers.append(
                    CapsuleType[routing](
                        name="CapsuleLayer%d" % i,
                        in_capsules = ((6*6 * channels) if i == 1 else layers[i-1]), 
                        in_dim = (dim if i == 1 else dimensions[i-1]), 
                        out_capsules = layers[i], 
                        out_dim = dimensions[i], 
                        use_bias = self.use_bias)
                )   

            if self.use_reconstruction:
                self.reconstruction_network = ReconstructionNetwork(
                    name="ReconstructionNetwork",
                    in_capsules=self.num_classes, 
                    in_dim=dimensions[-1])
            self.norm = Norm()


    def call(self, x, y):
        x = self.reshape(x)
        x = self.conv_1(x)
        x = self.primary(x)
        layers = [x]
        for capsule in self.capsule_layers:
            x = capsule(x)
            layers.append(x)
        r = self.reconstruction_network(x, y) if self.use_reconstruction else None
        out = self.norm(x)

        return out, r, layers