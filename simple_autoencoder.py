# An undercomplete autoencoder on MNIST dataset
from __future__ import division, print_function, absolute_import
import tensorflow.contrib.layers as lays
import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 500  # Number of samples in each batch
epoch_num = 5     # Number of epochs to train the network
lr = 0.001        # Learning rate


def save_plot(data, title):
    """
    Save images to file on the out folder
    """
    if not os.path.exists('out/'):
        os.makedirs('out/')

    fig = plt.figure()

    for i in range(50):
        plt.subplot(5, 10, i + 1)
        plt.imshow(data[i, ..., 0], cmap='gray')

    fig.savefig('out/' + title)
    plt.close(fig)


def resize_batch(imgs):
    """
    A function to resize a batch of MNIST images to (32, 32)
    Args:
       imgs: a numpy array of size [batch_size, 28 X 28].
    Returns:
        a numpy array of size [batch_size, 32, 32].
    """
    imgs = imgs.reshape((-1, 28, 28, 1))
    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
    return resized_imgs

# default aplha is 0.2, 0.01 works best for this example
# Function from TensorFlow v1.4 for backwards compatability


def leaky_relu(features, alpha=0.01, name=None):
    with ops.name_scope(name, "LeakyRelu", [features, alpha]):
        features = ops.convert_to_tensor(features, name="features")
        alpha = ops.convert_to_tensor(alpha, name="alpha")

        return math_ops.maximum(alpha * features, features)


def fully_connected(x, output_shape):
    shape = x.get_shape().as_list()
    dim = np.prod(shape[1:])

    x = tf.reshape(x, [-1, dim])
    x = tf.layers.dense(x, output_shape, activation=None)

    return x


def decoder(h, n, img_dim, channel_dim):
    h = tf.layers.dense(h, img_dim * img_dim * n, activation=None)
    h = tf.reshape(h, (-1, img_dim, img_dim, n))

    conv1 = tf.layers.conv2d(
        h, n, 3, padding="same", activation=leaky_relu)
    conv1 = tf.layers.conv2d(
        conv1, n, 3, padding="same", activation=leaky_relu)

    upsample1 = tf.image.resize_nearest_neighbor(
        conv1, size=(img_dim * 2, img_dim * 2))

    upsample_h1 = tf.image.resize_nearest_neighbor(
        h, size=(img_dim * 2, img_dim * 2))

    # concat skip connection: h -> first upsample
    upsample_concat1 = tf.concat([upsample_h1, upsample1], 3)

    conv2 = tf.layers.conv2d(
        upsample_concat1, n, 3, padding="same", activation=leaky_relu)
    conv2 = tf.layers.conv2d(
        conv2, n, 3, padding="same", activation=leaky_relu)

    upsample2 = tf.image.resize_nearest_neighbor(
        conv2, size=(img_dim * 4, img_dim * 4))

    # concat skip connection: h -> second upsample
    upsample_h2 = tf.image.resize_nearest_neighbor(
        h, size=(img_dim * 4, img_dim * 4))

    upsample_concat2 = tf.concat([upsample_h2, upsample2], 3)

    conv3 = tf.layers.conv2d(
        upsample_concat2, n, 3, padding="same", activation=leaky_relu)
    conv3 = tf.layers.conv2d(
        conv3, n, 3, padding="same", activation=leaky_relu)

    conv4 = tf.layers.conv2d(conv3, channel_dim, 3,
                             padding="same", activation=None)

    return conv4


def encoder(images, n, z_dim, channel_dim):
    conv1 = tf.layers.conv2d(
        images, n, 3, padding="same", activation=leaky_relu)

    conv2 = tf.layers.conv2d(
        conv1, n, 3, padding="same", activation=leaky_relu)
    conv2 = tf.layers.conv2d(
        conv2, n * 2, 3, padding="same", activation=leaky_relu)

    subsample1 = tf.layers.conv2d(
        conv2, n * 2, 3, strides=2, padding='same')

    conv3 = tf.layers.conv2d(subsample1, n * 2, 3,
                             padding="same", activation=leaky_relu)
    conv3 = tf.layers.conv2d(
        conv3, n * 3, 3, padding="same", activation=leaky_relu)

    subsample2 = tf.layers.conv2d(
        conv3, n * 3, 3, strides=2, padding='same')

    conv4 = tf.layers.conv2d(subsample2, n * 3, 3,
                             padding="same", activation=leaky_relu)
    conv4 = tf.layers.conv2d(
        conv4, n * 3, 3, padding="same", activation=leaky_relu)

    h = fully_connected(conv4, z_dim)  # z_dim aka embedding

    return h


def autoencoder(inputs):
    x = encoder(inputs, 32, 128, 3)
    x = decoder(x, 32, 32 // 4, 3)

    return x


# read MNIST dataset
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# calculate the number of batches per epoch
batch_per_ep = mnist.train.num_examples // batch_size

# input to the network (MNIST images)
ae_inputs = tf.placeholder(tf.float32, (None, 32, 32, 1))
ae_outputs = autoencoder(ae_inputs)  # create the Autoencoder network

# calculate the loss and optimize the network
# claculate the mean square error loss
loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs))
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# initialize the network
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for ep in range(epoch_num):  # epochs loop
        for batch_n in range(batch_per_ep):  # batches loop
            batch_img, batch_label = mnist.train.next_batch(
                batch_size)  # read a batch
            # reshape each sample to an (28, 28) image
            batch_img = batch_img.reshape((-1, 28, 28, 1))
            # reshape the images to (32, 32)
            batch_img = resize_batch(batch_img)
            _, c = sess.run([train_op, loss], feed_dict={ae_inputs: batch_img})
            print('Epoch: {} - cost= {:.5f}'.format((ep + 1), c))

    # test the trained network
    batch_img, batch_label = mnist.test.next_batch(50)
    batch_img = resize_batch(batch_img)
    recon_img = sess.run([ae_outputs], feed_dict={ae_inputs: batch_img})[0]

    save_plot(recon_img, 'Reconstructed Images.png')
    save_plot(batch_img, 'Input Images.png')
