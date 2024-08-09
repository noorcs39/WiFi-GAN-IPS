import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Load the CSV files into DataFrames
df_large = pd.read_csv('RSSISensors_Large.csv')
df_medium = pd.read_csv('RSSISensors_Medium.csv')
df_small = pd.read_csv('RSSISensors_Small.csv')

# Combine the datasets for a more comprehensive model
df_combined = pd.concat([df_large, df_medium, df_small])

# Drop the index column (Unnamed: 0) if present
df_combined = df_combined.loc[:, ~df_combined.columns.str.contains('^Unnamed')]

# Remove leading spaces from the column names
df_combined.columns = df_combined.columns.str.strip()

# Normalize the RSSI values and position coordinates
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_combined), columns=df_combined.columns)

# Separate features (RSSI values) and labels (coordinates)
X = df_normalized[['r1', 'r2', 'r3', 'r4']].values
y = df_normalized[['x', 'y']].values

# Define the Generator model
def build_generator(latent_dim, output_dim):
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, activation='relu', input_dim=latent_dim))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(output_dim, activation='tanh'))  # Output layer with tanh activation
    return model

# Define the Discriminator model
def build_discriminator(input_dim):
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim=input_dim))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation for binary classification
    return model

# Initialize the models
latent_dim = 100  # Dimensionality of the latent space
output_dim = X.shape[1]  # Number of features to generate (same as input features)

generator = build_generator(latent_dim, output_dim)
discriminator = build_discriminator(output_dim)
discriminator.compile(loss='binary_crossentropy', optimizer='adam')

# Combine the models into a GAN
gan = tf.keras.Sequential([generator, discriminator])
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Display summaries
generator.summary()
discriminator.summary()


# Function to generate random noise as input for the generator
def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# Function to generate and plot synthetic data
def generate_fake_samples(generator, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = generator.predict(x_input)
    y = np.zeros((n_samples, 1))
    return X, y
import matplotlib.pyplot as plt

# Train the GAN with loss tracking
def train_gan(generator, discriminator, gan, X, latent_dim, n_epochs=10000, n_batch=64):
    half_batch = int(n_batch / 2)
    
    # Lists to hold the loss values
    d_losses = []
    g_losses = []
    
    for i in range(n_epochs):
        # Train Discriminator on real samples
        idx = np.random.randint(0, X.shape[0], half_batch)
        X_real, y_real = X[idx], np.ones((half_batch, 1))
        d_loss_real = discriminator.train_on_batch(X_real, y_real)
        
        # Train Discriminator on fake samples
        X_fake, y_fake = generate_fake_samples(generator, latent_dim, half_batch)
        d_loss_fake = discriminator.train_on_batch(X_fake, y_fake)
        
        # Prepare points in latent space as input for the generator
        X_gan = generate_latent_points(latent_dim, n_batch)
        y_gan = np.ones((n_batch, 1))
        
        # Train the generator via the discriminator's error
        g_loss = gan.train_on_batch(X_gan, y_gan)
        
        # Record losses
        d_losses.append(d_loss_real + d_loss_fake)
        g_losses.append(g_loss)
        
        # Print losses every 1000 epochs
        if i % 1000 == 0:
            print(f"{i} [D loss: {d_loss_real + d_loss_fake}] [G loss: {g_loss}]")
    
    return d_losses, g_losses

# Train the GAN model and capture the loss history
d_losses, g_losses = train_gan(generator, discriminator, gan, X, latent_dim)

# Plot the losses of the Discriminator and Generator
plt.figure(figsize=(10, 5))
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.title('GAN Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()



# Generate synthetic data
n_samples = 100  # Number of synthetic samples to generate
synthetic_data, _ = generate_fake_samples(generator, latent_dim, n_samples)

# Plot the synthetic data
plt.figure(figsize=(10, 5))
plt.scatter(X[:, 0], X[:, 1], color='blue', label='Real Data', alpha=0.5)
plt.scatter(synthetic_data[:, 0], synthetic_data[:, 1], color='red', label='Synthetic Data', alpha=0.5)
plt.title('Real vs Synthetic Data (First Two RSSI Values)')
plt.xlabel('r1')
plt.ylabel('r2')
plt.legend()
plt.show()


