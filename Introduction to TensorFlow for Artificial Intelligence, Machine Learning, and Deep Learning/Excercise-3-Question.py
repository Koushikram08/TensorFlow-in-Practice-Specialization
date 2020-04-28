#!/usr/bin/env python
# coding: utf-8

# ## Exercise 3
# In the videos you looked at how you would improve Fashion MNIST using Convolutions. For your exercise see if you can improve MNIST to 99.8% accuracy or more using only a single convolutional layer and a single MaxPooling 2D. You should stop training once the accuracy goes above this amount. It should happen in less than 20 epochs, so it's ok to hard code the number of epochs for training, but your training must end once it hits the above metric. If it doesn't, then you'll need to redesign your layers.
# 
# I've started the code for you -- you need to finish it!
# 
# When 99.8% accuracy has been hit, you should print out the string "Reached 99.8% accuracy so cancelling training!"

import tensorflow as tf
from os import path, getcwd, chdir

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.998):
            print("\nReached 99.8% accuracy so cancelling training!")
            self.model.stop_training = True

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def train_mnist_conv():
       mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data(path=path)    
    training_images=training_images.reshape(60000, 28, 28, 1)
    training_images  = training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images / 255.0
    callbacks = myCallback()
    

    model = tf.keras.models.Sequential([  
            tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(input_shape=(28,28)),
            tf.keras.layers.Dense(512,activation=tf.nn.relu),
            tf.keras.layers.Dense(10,activation=tf.nn.softmax)
            ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model fitting
    history = model.fit(
         training_images,training_labels, epochs=10, callbacks=[callbacks]
    )
    return history.epoch, history.history['acc'][-1]

_, _ = train_mnist_conv()

