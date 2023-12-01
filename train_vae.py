# train_vae.py
import numpy as np
import tensorflow as tf
from vae_model import VariationalAutoencoder

# Load and preprocess the Fashion-MNIST dataset
(x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# Instantiate the VAE model
latent_dim = 2  # Set your desired latent dimension
vae = VariationalAutoencoder(latent_dim)

# Compile the model
vae.compile(optimizer="adam", loss="binary_crossentropy")

# Train the model
vae.fit(x_train, x_train, epochs=10, batch_size=128, validation_data=(x_test, x_test))

# Generate fashion samples using the trained VAE
num_samples = 10
random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))
generated_fashions = vae.decoder.predict(random_latent_vectors)

# Save the model if needed
# vae.save("vae_fashion_model.h5")