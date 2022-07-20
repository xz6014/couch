import torch
import torch.nn as nn

class ContactVAEEncoder(nn.Module):
    def __init__(self, input_dim, cond_dim, p_dim, h_dim, z_dim):
        super(ContactVAEEncoder, self).__init__()
        self.cond_dim = cond_dim
        self.p_dim = p_dim

        self.cond_encoder1 = nn.Sequential(
            nn.Linear(cond_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 64))

        self.main = nn.Sequential(
            nn.Linear(input_dim + 64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16))

        self.fc1 = nn.Linear(16, z_dim)
        self.fc2 = nn.Linear(16, z_dim)

    def forward(self, y, cond):
        cond1 = cond[:, -self.cond_dim:]
        cond1 = self.cond_encoder1(cond1)

        x = torch.cat((y, cond1), dim=1)
        x = self.main(x)
        return self.fc1(x), self.fc2(x)


class ContactVAEDecoder(nn.Module):
    def __init__(self, input_dim, cond_dim, p_dim, h_dim, z_dim):
        super(ContactVAEDecoder, self).__init__()
        self.cond_dim = cond_dim
        self.p_dim = p_dim
        self.z_dim = z_dim
        self.cond_encoder1 = nn.Sequential(
            nn.Linear(cond_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 64))


        self.main = nn.Sequential(
            nn.Linear(z_dim + 64 , 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, input):
        z = input[:, :self.z_dim]

        cond1 = input[:, -self.cond_dim:]

        cond1 = self.cond_encoder1(cond1)
        z = torch.cat((z, cond1), dim=1)
        return self.main(z)


class ContactVAE(nn.Module):
    def __init__(self, input_dim_goalnet, interaction_dim, pose_dim, h_dim_goalnet, z_dim_goalnet, **kwargs):
        super(ContactVAE, self).__init__()
        self.encoder = ContactVAEEncoder(input_dim_goalnet, interaction_dim, pose_dim, h_dim_goalnet, z_dim_goalnet)
        self.decoder = ContactVAEDecoder(input_dim_goalnet, interaction_dim, pose_dim, h_dim_goalnet, z_dim_goalnet)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, y, cond):
        mu, logvar = self.encoder(y, cond)

        z = self.reparameterize(mu, logvar)

        decoder_input = torch.cat([z, cond], -1)
        return self.decoder(decoder_input), mu, logvar