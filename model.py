import torch
import torch.nn as nn
from attention import Attention


class AadNet(torch.nn.Module):
    def __init__(self, input_param, dropout, device, config):
        super(AadNet, self).__init__()
        self.input_dim = len(input_param)

        self.fc_params = nn.Sequential(
            nn.Linear(len(input_param), 1024),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 64),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )

        self.attention = Attention(64, num_heads=64, dropout=dropout)

        self.albedo_predictor = nn.Sequential(
            nn.Linear(64, 1024),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )

        self.diameter_predictor = nn.Sequential(
            nn.Linear(64 + 1, 1024),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )

        self.albedo_mean = torch.tensor(config.albedo_mean, dtype=torch.float32).to(device)
        self.albedo_std = torch.tensor(config.albedo_std, dtype=torch.float32).to(device)
        self.diameter_mean = torch.tensor(config.diameter_mean, dtype=torch.float32).to(device)
        self.diameter_std = torch.tensor(config.diameter_std, dtype=torch.float32).to(device)
        self.h_mean = torch.tensor(config.h_mean, dtype=torch.float32).to(device)
        self.h_std = torch.tensor(config.h_std, dtype=torch.float32).to(device)

    def forward(self, x_params):
        x = self.fc_params(x_params)
        x = x.unsqueeze(1)
        x_attn = self.attention(x)
        x_attn = x_attn.squeeze(1)
        albedo = self.albedo_predictor(x_attn)
        albedo_denorm = torch.abs(albedo * self.albedo_std + self.albedo_mean)
        h = x_params[:, 0]
        h_denorm = h * self.h_std + self.h_mean
        calculated_diameter = 1329 / torch.sqrt(albedo_denorm.squeeze(1) + 1e-10) * 10 ** (-h_denorm / 5)
        norm_diameter = (calculated_diameter - self.diameter_mean) / self.diameter_std
        x_diameter_input = torch.cat([x_attn, norm_diameter.unsqueeze(1)], dim=1)
        diameter = self.diameter_predictor(x_diameter_input)

        return albedo, diameter
