# pyrefly: ignore [missing-import]
import torch.nn as nn
import torch
import yaml
import os

class SemanticProjector(nn.Module):
    def __init__(self):
        super().__init__()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.abspath(os.path.join(current_dir, '..', 'config.yaml'))
        if not os.path.exists(config_path):
            config_path = os.path.abspath(os.path.join(current_dir, '../../..', 'config.yaml'))
        if not os.path.exists(config_path):
            if os.path.exists('config.yaml'):
                config_path = 'config.yaml'
            elif os.path.exists('S2ST/models/s2st/config.yaml'):
                config_path = 'S2ST/models/s2st/config.yaml'

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.config = self.config['semantic_projector']

        self.input_dim = self.config['input_dim']
        self.hidden_dim= self.config['hidden_dim']
        self.output_dim= self.config['output_dim']
        self.dropout   = self.config['dropout']
        self.activation= self.config['activation']
        self.num_layer = self.config['layers']
        self.normalize = self.config['normalize']

        if self.activation == 'ReLU':
            self.ac_fnc = nn.ReLU()
        else:
            self.ac_fnc = nn.GELU()

        self.block1 = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            self.get_norm(self.output_dim),
            self.ac_fnc,
            nn.Dropout(self.dropout)
        ) ### batch x seq x 1024 -> batch x seq x 768
        
        self.block2 = nn.Sequential(
            nn.Linear(self.output_dim, self.hidden_dim),
            self.get_norm(self.hidden_dim),
            self.ac_fnc,
            nn.Dropout(self.dropout)
        ) ### batch x seq x 768 -> batch x seq x 512

        self.block3 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
            self.get_norm(self.output_dim),
            self.ac_fnc,
            nn.Dropout(self.dropout)
        ) ### batch x seq x 512 -> batch x seq x 768

    def get_norm(self, hidden_dim):
        if self.normalize:
            return nn.LayerNorm(hidden_dim)
        return nn.Identity()

    def forward(self, x: torch.tensor):
        ### x: batch x seq x 1024

        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2) + x1

        return x3
