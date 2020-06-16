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
from capsule.utils import margin_loss
from data.mnist import create_mnist
from data.fashion_mnist import create_fashion_mnist


#
# Hyperparameters and cmd args
#
argparser = argparse.ArgumentParser(description="Show limitations of capsule networks")
argparser.add_argument("--learning_rate", default=0.0001, type=float, 
  help="Learning rate of adam")
argparser.add_argument("--reconstruction_weight", default=0.0005, type=float, 
  help="Learning rate of adam")
argparser.add_argument("--log_dir", default="experiments/tmp", 
  help="Learning rate of adam")    
argparser.add_argument("--batch_size", default=256, type=int, 
  help="Learning rate of adam")
argparser.add_argument("--enable_tf_function", default=True, type=bool, 
  help="Enable tf.function for faster execution")
argparser.add_argument("--epochs", default=30, type=int, 
  help="Defines the number of epochs to train the network")
argparser.add_argument("--use_bias", default=False, type=bool, 
  help="Add a bias term to the preactivation")
argparser.add_argument("--use_reconstruction", default=True, type=bool, 
  help="Use the reconstruction network as regularization loss")
argparser.add_argument("--test", default=True, type=bool, 
  help="Run tests after each epoch?")

# Architecture
argparser.add_argument("--dataset", default="mnist",
  help="mnist, fashion_mnist")
argparser.add_argument("--routing", default="rba",
  help="rba, em")
argparser.add_argument("--layers", default="32,16,16,10",
  help=", spereated list of layers. Each number represents the number of hidden units except for the first layer the number of channels.")
argparser.add_argument("--dimensions", default="8,12,12,16",
  help=", spereated list of layers. Each number represents the dimension of the layer.")

# Load hyperparameters from cmd args and update with json file
args = argparser.parse_args()


def compute_loss(logits, y, reconstruction, x):
  """ The loss is the sum of the margin loss and the reconstruction loss 
      as defined in [2]
  """ 
  num_classes = tf.shape(logits)[1]

  # Calculate margin loss
  loss = margin_loss(logits, tf.one_hot(y, num_classes))
  loss = tf.reduce_mean(loss)

  # Calculate reconstruction loss
  if args.use_reconstruction:
    x_1d = tf.keras.layers.Flatten()(x)
    distance = tf.square(reconstruction - x_1d)
    reconstruction_loss = tf.reduce_sum(distance, axis=-1)
    reconstruction_loss = args.reconstruction_weight * tf.reduce_mean(reconstruction_loss)
  else:
    reconstruction_loss = 0

  loss = loss + reconstruction_loss

  return loss, reconstruction_loss


def train(train_ds, test_ds, class_names):
  """ Train capsule networks mirrored on multiple gpu's
  """

  # Run training for multiple epochs mirrored on multiple gpus
  strategy = tf.distribute.MirroredStrategy()
  num_replicas = strategy.num_replicas_in_sync

  (batch_train_ds, online_train_ds) = train_ds
  batch_train_ds = strategy.experimental_distribute_dataset(batch_train_ds)
  online_train_ds = strategy.experimental_distribute_dataset(online_train_ds)
  test_ds = strategy.experimental_distribute_dataset(test_ds)

  # Create a checkpoint directory to store the checkpoints.
  ckpt_dir = os.path.join(args.log_dir, "ckpt/", "ckpt")

  train_writer = tf.summary.create_file_writer("%s/log/train" % args.log_dir)
  test_writer = tf.summary.create_file_writer("%s/log/test" % args.log_dir)

  with strategy.scope():
    layers = list(map(int, args.layers.split(","))) if args.layers != "" else []
    dimensions = list(map(int, args.dimensions.split(","))) if args.dimensions != "" else []

    model = CapsNet(routing=args.routing, layers=layers, dimensions=dimensions, 
      use_bias=args.use_bias, use_reconstruction=args.use_reconstruction)
    optimizer = optimizers.Adam(learning_rate=args.learning_rate)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    # Define metrics 
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    
    # Function for a single training step
    def train_step(inputs):
      x, y = inputs
      with tf.GradientTape() as tape:
        logits, reconstruction, layers = model(x, y)
        loss, _ = compute_loss(logits, y, reconstruction, x)
      
      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      train_accuracy.update_state(y, logits)
      return loss, (x, reconstruction)

    # Function for a single test step
    def test_step(inputs):
      x, y = inputs
      logits, reconstruction, _ = model(x, y)
      loss, _ = compute_loss(logits, y, reconstruction, x)
      
      test_accuracy.update_state(y, logits)
      test_loss.update_state(loss)

      pred = tf.math.argmax(logits, axis=1)
      cm = tf.math.confusion_matrix(y, pred, num_classes=10)
      return cm

    # Define functions for distributed training
    def distributed_train_step(dataset_inputs):
      return strategy.experimental_run_v2(train_step, args=(dataset_inputs,))

    def distributed_test_step(dataset_inputs):
      return strategy.experimental_run_v2(test_step, args=(dataset_inputs, ))
    
    if args.enable_tf_function:
      distributed_train_step = tf.function(distributed_train_step)
      distributed_test_step = tf.function(distributed_test_step)

    # Loop for multiple epochs
    step = 0
    for epoch in range(args.epochs):
      ########################################
      # Test
      ########################################
      if args.test:
        cm = np.zeros((10, 10))
        for data in test_ds:
          distr_cm = distributed_test_step(data)
          for r in range(num_replicas):
            cm += distr_cm.values[r]

        # Log test results (for replica 0 only for activation map and reconstruction)
        figure = utils.plot_confusion_matrix(cm.numpy(), class_names)
        cm_image = utils.plot_to_image(figure)
        print("TEST | epoch %d (%d): acc=%.4f, loss=%.4f" % 
              (epoch, step, test_accuracy.result(), test_loss.result()), flush=True)  

        with test_writer.as_default(): 
          tf.summary.image("Confusion Matrix", cm_image, step=step)
          tf.summary.scalar("General/Accuracy", test_accuracy.result(), step=step)
          tf.summary.scalar("General/Loss", test_loss.result(), step=step)
        test_accuracy.reset_states()
        test_loss.reset_states()
        test_writer.flush()


      ########################################
      # Train
      ########################################
      for data in batch_train_ds:
        start = time.time()
        distr_loss, distr_imgs = distributed_train_step(data)
        train_loss = tf.reduce_mean(distr_loss.values)

        # Logging
        if step % 100 == 0:
          time_per_step = (time.time()-start) * 1000 / 100
          print("TRAIN | epoch %d (%d): acc=%.4f, loss=%.4f | Time per step[ms]: %.2f" % 
              (epoch, step, train_accuracy.result(), train_loss.numpy(), time_per_step), flush=True)     

          # Create some recon tensorboard images (only GPU 0)
          if args.use_reconstruction:
            x, recon_x = distr_imgs[0].values[0], distr_imgs[1].values[0]
            recon_x = tf.reshape(recon_x, [-1, tf.shape(x)[1], tf.shape(x)[2]])  
            img = tf.concat([x, recon_x], axis=1)
            img = tf.expand_dims(img, -1)
            with train_writer.as_default():
              tf.summary.image(
                "X & XAdv & Recon",
                img,
                step=step,
                max_outputs=3,)

          with train_writer.as_default(): 
            # Write scalars
            tf.summary.scalar("General/Accuracy", train_accuracy.result(), step=step)
            tf.summary.scalar("General/Loss", train_loss.numpy(), step=step)

          train_accuracy.reset_states()
          start = time.time()

          train_writer.flush()
        
        step += 1


      ####################
      # Checkpointing
      if epoch % 5 == 0:
        checkpoint.save(ckpt_dir)



#
# M A I N
#
def main():
  # Configurations for cluster
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  for r in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[r], True)

  # Write log folder and arguments
  if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

  with open("%s/args.txt" % args.log_dir, "w") as file:
     file.write(json.dumps(vars(args)))

  # Load data
  if args.dataset=="mnist":
    train_ds, test_ds, class_names = create_mnist(args.batch_size)
  else:
    train_ds, test_ds, class_names = create_fashion_mnist(args.batch_size)

  # Train capsule network
  train(train_ds, test_ds, class_names)


       
if __name__ == '__main__':
    main()