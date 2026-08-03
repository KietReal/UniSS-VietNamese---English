import yaml
import torch.nn as nn
import torch
import os

class AudioHeader(nn.Module):
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

        self.config = self.config['audio_header']
        self.input_dim = self.config['input_dim']
        self.output_dim= self.config['output_dim']

        ### batch x seq x 768 -> batch x seq x 8194
        self.audio_head = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x: torch.tensor):
        
        return self.audio_head(x)