# generate_fashions.py
import numpy as np
import matplotlib.pyplot as plt
from vae_model import VariationalAutoencoder

# Load the trained VAE model
vae = VariationalAutoencoder(latent_dim=2)
vae.load_weights("path/to/your/vae_model_weights.h5")

# Generate fashion samples using the trained VAE
num_samples = 10
random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))
generated_fashions = vae.decoder.predict(random_latent_vectors)

# Display the generated fashions using matplotlib
plt.figure(figsize=(15, 3))
for i in range(num_samples):
    # Original Image
    plt.subplot(2, num_samples, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

    # Generated Image
    plt.subplot(2, num_samples, i + 1 + num_samples)
    plt.imshow(generated_fashions[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

plt.show()