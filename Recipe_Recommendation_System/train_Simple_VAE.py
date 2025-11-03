import torch
import csv
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
from Simple_VAE import WAE

data = np.load('input_data/vae_input_data.npz')
text_embeds = data['text_embeds']
numeric_scaled = data['numeric_scaled']
print("Data loaded successfully")

# Dimensions
numeric_dim = numeric_scaled.shape[1]
text_dim = text_embeds.shape[1]
latent_dim = 32  # choose latent size

# Loss function
def wae_loss(numeric_recon, numeric, text_recon, text, mu, logvar, alpha=1.0):
    # Reconstruction
    numeric_loss = nn.MSELoss()(numeric_recon, numeric)
    text_loss = nn.MSELoss()(text_recon, text)
    recon_loss = numeric_loss + text_loss
    
    # KL loss (or Wasserstein regularization)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / numeric.size(0)
    
    return recon_loss + alpha * kl_loss


# Training loop (example)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = WAE(numeric_dim, text_dim, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

numeric_tensor = torch.tensor(numeric_scaled, dtype=torch.float32).to(device)
text_tensor = torch.tensor(text_embeds, dtype=torch.float32).to(device)

losses = []
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    numeric_recon, text_recon, mu, logvar = model(numeric_tensor, text_tensor)
    loss = wae_loss(numeric_recon, numeric_tensor, text_recon, text_tensor, mu, logvar)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# Save losses to CSV
with open('output_loss/training_losses.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'loss'])
    for i, l in enumerate(losses):
        writer.writerow([i, l])

print("Losses saved to output_loss/training_losses.csv")


model.eval()
with torch.no_grad():
    z_all, _, _ = model.encoder(numeric_tensor, text_tensor)

z_all = z_all.cpu().numpy()  # shape: [num_recipes, latent_dim]
np.save('output_embeds/latent_recipes.npy', z_all)
print("Latent vectors saved to latent_recipes.npy")