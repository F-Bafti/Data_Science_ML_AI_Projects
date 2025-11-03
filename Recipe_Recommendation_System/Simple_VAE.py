import torch
import torch.nn as nn
import torch.optim as optim

# Encoder
class Encoder(nn.Module):
    def __init__(self, numeric_dim, text_dim, latent_dim):
        super().__init__()
        # Numeric branch
        self.numeric_branch = nn.Sequential(
            nn.Linear(numeric_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # Text branch
        self.text_branch = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        # Merge
        self.fc_mu = nn.Linear(32 + 32, latent_dim)
        self.fc_logvar = nn.Linear(32 + 32, latent_dim)
        
    def forward(self, numeric, text):
        n = self.numeric_branch(numeric)
        t = self.text_branch(text)
        h = torch.cat([n, t], dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar
    


    # Decoder
class Decoder(nn.Module):
    def __init__(self, numeric_dim, text_dim, latent_dim):
        super().__init__()
        # Shared latent to numeric
        self.numeric_decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, numeric_dim)
        )
        # Shared latent to text
        self.text_decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, text_dim)
        )
        
    def forward(self, z):
        numeric_recon = self.numeric_decoder(z)
        text_recon = self.text_decoder(z)
        return numeric_recon, text_recon
    

    # Full VAE
class WAE(nn.Module):
    def __init__(self, numeric_dim, text_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(numeric_dim, text_dim, latent_dim)
        self.decoder = Decoder(numeric_dim, text_dim, latent_dim)
    
    def forward(self, numeric, text):
        z, mu, logvar = self.encoder(numeric, text)
        numeric_recon, text_recon = self.decoder(z)
        return numeric_recon, text_recon, mu, logvar