
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras



def create_cifar(batch_size):
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
    train_images = train_images.astype(np.float32)
    train_images = train_images / 255.0
    train_images = tf.image.per_image_standardization(train_images)
    train_images = tf.image.resize(train_images, [28,28])

    train_images = tf.squeeze(train_images)
    train_labels = train_labels.astype(np.int64)
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    batch_train_ds = train_ds.shuffle(60000).batch(batch_size)
    online_train_ds = train_ds.shuffle(60000).batch(1)

    test_images = test_images.astype(np.float32)
    test_images = test_images / 255.0
    test_images = tf.image.per_image_standardization(test_images)
    test_images = tf.image.resize(test_images, [28,28])
    test_labels = test_labels.astype(np.int64)
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_ds = test_ds.batch(100)

    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    return (batch_train_ds, online_train_ds), test_ds, class_names