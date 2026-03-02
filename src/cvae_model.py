import torch
import torch.nn as nn

class ObjectEncoder(nn.Module):
    def __init__(self, input_dim=7869, hidden_dims=[512, 256], out_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[1], out_dim)
        )
    def forward(self, x):
        return self.fc(x)

class HandEncoder(nn.Module):
    def __init__(self, input_dim=114, hidden_dim=128, out_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.fc(x)

class PriorNet(nn.Module):
    def __init__(self, cond_dim=128, latent_dim=64):
        super().__init__()
        self.fc_mu = nn.Linear(cond_dim, latent_dim)
        self.fc_logvar = nn.Linear(cond_dim, latent_dim)
    def forward(self, c):
        return self.fc_mu(c), self.fc_logvar(c)

class PosteriorNet(nn.Module):
    def __init__(self, cond_dim=128, hand_dim=64, latent_dim=64):
        super().__init__()
        self.fc_mu = nn.Linear(cond_dim + hand_dim, latent_dim)
        self.fc_logvar = nn.Linear(cond_dim + hand_dim, latent_dim)
    def forward(self, h, c):
        x = torch.cat([h, c], dim=-1)
        return self.fc_mu(x), self.fc_logvar(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=64, cond_dim=128, hidden_dims=[256, 512], output_dim=114):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[1], output_dim)
        )
    def forward(self, z, c):
        x = torch.cat([z, c], dim=-1)
        return self.fc(x)

class CVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.object_encoder = ObjectEncoder()
        self.hand_encoder = HandEncoder()
        self.prior_net = PriorNet(cond_dim=128, latent_dim=latent_dim)
        self.posterior_net = PosteriorNet(cond_dim=128, hand_dim=64, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, obj, hand):
        c = self.object_encoder(obj)
        h = self.hand_encoder(hand)
        mu_q, logvar_q = self.posterior_net(h, c)
        mu_p, logvar_p = self.prior_net(c)
        z = self.reparameterize(mu_q, logvar_q)
        recon_hand = self.decoder(z, c)
        return recon_hand, mu_q, logvar_q, mu_p, logvar_p

def kl_divergence(mu_q, logvar_q, mu_p, logvar_p):
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * torch.sum(logvar_p - logvar_q - 1 + (var_q + (mu_q - mu_p)**2) / var_p, dim=1)
    return torch.mean(kl)

if __name__ == '__main__':
    print('The CVAE 04 is loaded.')
