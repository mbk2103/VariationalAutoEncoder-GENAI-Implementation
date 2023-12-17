# VariationalAutoEncoder-GENAI-Implementation

# Variational Autoencoder (VAE) - GENAI Implementation

This project implements a Variational Autoencoder (VAE) for generating fashion samples using the Fashion-MNIST dataset. The VAE is trained on the dataset, and generated fashion samples can be visualized using the trained model.

## Project Structure

The project is organized into several files:

- **main.py**: Entry point for running the VAE training and fashion sample generation.
- **train_vae.py**: Script for training the VAE model on the Fashion-MNIST dataset.
- **vae_model.py**: Implementation of the Variational Autoencoder model.
- **generate_fashions.py**: Script for generating and visualizing fashion samples using the trained VAE.

## Usage

To train the VAE model and generate fashion samples, run the following commands:

```bash
python main.py
```

This will execute the training script and generate fashion samples using the trained model.

### Model Configuration
- Latent Dimension: Set your desired latent dimension.
- Epochs: 10
- Batch Size: 128

## Dependencies
- TensorFlow
- NumPy
- Matplotlib

Install dependencies using: 
```bash
pip install tensorflow numpy matplotlib
```

Feel free to explore and modify the project to suit your needs!


Note: Make sure to replace `"path/to/your/vae_model_weights.h5"` with the actual path to the saved weights of your trained VAE model. Additionally, you may need to adjust other details based on your specific setup.

