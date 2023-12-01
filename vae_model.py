# vae_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class VariationalAutoencoder(models.Model):
    def __init__(self, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = self.build_encoder()

        # Decoder
        self.decoder = self.build_decoder()

    def build_encoder(self):
        input_img = layers.Input(shape=(28, 28, 1))
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(input_img)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)

        return models.Model(input_img, [z_mean, z_log_var])

    def build_decoder(self):
        input_z = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(7 * 7 * 64, activation="relu")(input_z)
        x = layers.Reshape((7, 7, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        decoded = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)

        return models.Model(input_z, decoded)

    def sample(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sample(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        return reconstructed