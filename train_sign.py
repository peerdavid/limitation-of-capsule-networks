try:
    import cluster_setup
except ImportError:
    pass

import os
import sys
import time
import math
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, optimizers, datasets
import sklearn.metrics

import utils
from capsule.capsule_network import CapsNet
from capsule.sign_capsule_network import SignCapsNet
from capsule.utils import margin_loss
from data.sign import create_sign_data


#
# Hyperparameters and cmd args
#
# Learning hyperparameters
argparser = argparse.ArgumentParser(description="Show limitations of capsule networks")
argparser.add_argument("--learning_rate", default=0.005, type=float, 
  help="Learning rate of adam")
argparser.add_argument("--batch_size", default=256, type=int, 
  help="Learning rate of adam")
argparser.add_argument("--epochs", default=50, type=int, 
  help="Defines the number of epochs to train the network")
argparser.add_argument("--log_dir", default="experiments/robust", 
  help="Learning rate of adam")    
argparser.add_argument("--enable_tf_function", default=True, type=bool, 
  help="Enable tf.function for faster execution")

# Routing properties
argparser.add_argument("--routing", default="em",
  help="rba, em")
argparser.add_argument("--use_bias", default=False, type=bool, 
  help="Add a bias term to the preactivation")

# Dataset properties
argparser.add_argument("--dataset_size", default=20000, type=int, 
  help="Size of training set")

args = argparser.parse_args()


def compute_loss(logits, y):
  """ The loss is the sum of the margin loss and the reconstruction loss 
      as defined in [2], no reconstruciton loss for lines
  """ 
  # Calculate margin loss
  loss = margin_loss(logits, tf.one_hot(y, 2), down_weighting=1.0)
  loss = tf.reduce_mean(loss)
  loss = loss

  return loss


def compute_accuracy(logits, labels):
    predictions = tf.cast(tf.argmax(logits, axis=1), tf.int32)
    return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))


def train(train_ds):
  """ Train capsule networks mirrored on multiple gpu's
  """

  # Run training for multiple epochs mirrored on multiple gpus
  strategy = tf.distribute.MirroredStrategy()
  train_ds = strategy.experimental_distribute_dataset(train_ds)
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

  # Create a checkpoint directory to store the checkpoints.
  ckpt_dir = os.path.join(args.log_dir, "ckpt/", "ckpt")
  train_writer = tf.summary.create_file_writer("%s/log/train" % args.log_dir)

  with strategy.scope():
    model = SignCapsNet(routing=args.routing, layers=[20, 2], use_bias=args.use_bias)
    optimizer = optimizers.Adam(learning_rate=args.learning_rate)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    
    # Function for a single training step
    def train_step(inputs):
      x, y = inputs
      with tf.GradientTape() as tape:
        logits = model(x, y)
        loss = compute_loss(logits, y)
      
      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      train_accuracy.update_state(y, logits)
      acc = compute_accuracy(logits, y)
      
      return loss

    # Define functions for distributed training
    def distributed_train_step(dataset_inputs):
      return strategy.experimental_run_v2(train_step, args=(dataset_inputs,))

    if args.enable_tf_function:
      distributed_train_step = tf.function(distributed_train_step)

    ########################################
    # Train
    ########################################
    step = 0
    for epoch in range(args.epochs):
      for data in train_ds:
        # Training step
        start = time.time()
        distr_train_loss = distributed_train_step(data)  
        train_loss = tf.reduce_mean(distr_train_loss.values)

        # Logging
        if step % 100 == 0:
          time_per_step = (time.time()-start) * 1000 / 100
          print("TRAIN | epoch %d (%d): acc=%.2f, loss=%.6f | Time per step[ms]: %.2f" % 
              (epoch, step, train_accuracy.result(), train_loss, time_per_step), flush=True)     

          with train_writer.as_default(): 
            tf.summary.scalar("General/Accuracy", train_accuracy.result(), step=step)
            tf.summary.scalar("General/Loss", train_loss, step=step)

          start = time.time()
          train_writer.flush()
          train_accuracy.reset_states()

        step += 1
      
      # Checkpointing
      if epoch == args.epochs-1:
        checkpoint.save(ckpt_dir)


#
# M A I N
#
def main():
  # Write log folder and arguments
  if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

  with open("%s/args.txt" % args.log_dir, "w") as file:
     file.write(json.dumps(vars(args)))

  # Load data
  train_ds = create_sign_data(
    batch_size = args.batch_size,
    dataset_size = args.dataset_size)

  # Train capsule network
  train(train_ds)


       
if __name__ == '__main__':
    main()