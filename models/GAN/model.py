import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, UpSampling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import numpy as np

def build_generator(latent_dim):
    inputs = Input(shape=(latent_dim,))
    x = Dense(128 * 7 * 7)(inputs)
    x = Reshape((7, 7, 128))(x)
    x = Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(1, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)
    outputs = x
    model = Model(inputs=inputs, outputs=outputs, name='generator')
    return model

def build_discriminator(img_shape):
    inputs = Input(shape=img_shape)
    x = UpSampling2D(size=(2, 2))(inputs)
    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs, name='discriminator')
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    inputs = Input(shape=(100,))
    x = generator(inputs)
    outputs = discriminator(x)
    model = Model(inputs=inputs, outputs=outputs, name='gan')
    return model