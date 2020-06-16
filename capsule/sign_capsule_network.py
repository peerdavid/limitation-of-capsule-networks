

import tensorflow as tf
from capsule.capsule_layer import Capsule
from capsule.em_capsule_layer import EMCapsule
from capsule.primary_capsule_layer import PrimaryCapsule
from capsule.reconstruction_network import ReconstructionNetwork
from capsule.norm_layer import Norm

layers = tf.keras.layers
models = tf.keras.models


class SignCapsNet(tf.keras.Model):

    def __init__(self, routing, layers, use_bias=False):
        super(SignCapsNet, self).__init__()

        CapsuleType = {
            "rba": Capsule,
            "em": EMCapsule
        }
        self.use_bias=use_bias
        self.capsule_layers = []
        
        in_capsule = 1
        for num_capsules in  layers:
            self.capsule_layers.append(
                CapsuleType[routing](
                    in_capsules=in_capsule, 
                    in_dim=1, out_capsules=num_capsules, 
                    out_dim=1, use_bias=self.use_bias,
                    stdev=0.5)
            )
            in_capsule = num_capsules 
        
        self.norm = Norm()


    def call(self, x, y):
        x = tf.expand_dims(x, axis=1)

        for capsule in self.capsule_layers:
            x = capsule(x)
        out = self.norm(x)
        return out
