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
argparser.add_argument("--batch_size", default=256, type=int, 
  help="Learning rate of adam")
argparser.add_argument("--epochs", default=10, type=int, 
  help="Defines the number of epochs to train the network") 
argparser.add_argument("--enable_tf_function", default=True, type=bool, 
  help="Enable tf.function for faster execution")
argparser.add_argument("--use_bias", default=False, type=bool, 
  help="Add a bias term to the preactivation")
argparser.add_argument("--logging", default=False, type=bool, 
  help="Detailed logging")

# Routing properties
argparser.add_argument("--routing", default="em",
  help="rba, em")

# Dataset properties
argparser.add_argument("--dataset_size", default=4096, type=int, 
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


def train(train_ds, learning_rate, layers, use_bias):
  """ Train capsule networks mirrored on multiple gpu's
  """

  # Initialize
  model = SignCapsNet(routing=args.routing, layers=layers, use_bias=use_bias)
  optimizer = optimizers.Adam(learning_rate=learning_rate)
  
  # Function for a single training step
  def train_step(inputs):
    x, y = inputs
    with tf.GradientTape() as tape:
      logits = model(x, y)
      loss = compute_loss(logits, y)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    acc = compute_accuracy(logits, y)
    return loss, acc
  
  def test_step(inputs):
    x, y = inputs
    logits = model(x, y)
    acc = compute_accuracy(logits, y)
    return acc

  if args.enable_tf_function:
    train_step = tf.function(train_step)
    test_step = tf.function(test_step)

  ########################################
  # Training
  ########################################
  step = 0
  for epoch in range(args.epochs):
    for data in train_ds:
      loss, acc = train_step(data) 

      # Logging
      if args.logging:
        print("TRAIN | epoch %d (%d): acc=%.2f, loss=%.6f" % 
          (epoch, acc, loss), 
          flush=True)   

  ########################################
  # Evaluate accuracy for all datapoints
  ########################################
  acc = np.mean([test_step(data) for data in train_ds])
  return acc


#
# M A I N
#
def main():

  # Load data
  train_ds = create_sign_data(
    batch_size = args.batch_size,
    dataset_size = args.dataset_size)

  ########################################
  # Architecture & hyperparameter search
  ########################################
  executions = []
  total_solved = 0
  for lr in [0.001, 0.005, 0.01]:
    for num_hidden_layers in [3,2,1]:
      for num_caps in [30,25,20,15,10,5]:
        for caps_dim in [30,25,20,15]:
          
          # Using the current architecture we try 3x if we can solve the problem
          num_retries = 3
          for i in range(num_retries):

            # Create architecture
            layers = [(num_caps, caps_dim) for i in range(num_hidden_layers)]

            # Train network
            acc = train(
              train_ds, 
              learning_rate=lr, 
              layers=layers, 
              use_bias=args.use_bias)

            # Evaluate solution
            solved = bool(acc > 0.6)
            total_solved += int(solved)
            print("lr=%.5f, num_layers=%d, num_caps=%d, caps_dim=%d | acc=%.3f | solved = %s" % (lr, 
              num_hidden_layers+1, num_caps, caps_dim, acc, solved), flush=True)
            
            if solved:
              break
          
          # Add accuracy of solved / not solved execution
          executions.append(acc)
  
  #
  # Log results
  #  
  print("\n==========================", flush=True)
  print("Accuracy | Num solved", flush=True)
  print("==========================", flush=True)
  for b in [0.5, 0.6, 0.7, 0.8, 0.9]:
    num_solved = np.sum([1 if e > b else 0 for e in executions])
    log = "> %.2f | %d" % (b, num_solved)
    print(log, flush=True)

    file_name = "experiments/routing_%s_bias_%s.txt" % (args.routing, args.use_bias)
    with open(file_name, 'a') as f:
      f.write(log)

  print("==========================")


       
if __name__ == '__main__':
    main()