import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal

class ContinuousActor(nn.Module):
    def __init__(self, deter_size, stoch_size, act_dim, info):
        super().__init__()

        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.act_dim = act_dim
        self.num_layers = info['num_layers']
        self.hid_dim = info['hid_dim']
        self.activation = info['activation']
        self.dist = info['dist']
        
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
        self.model = self._build_model()

    def _build_model(self):
        model = [nn.Linear(self.deter_size + self.stoch_size, self.hid_dim)]
        model += [self.activation()]

        for i in range(self.num_layers - 1):
            model += [nn.Linear(self.hid_dim, self.hid_dim)]
            model += [self.activation()]

        model += [nn.Linear(self.hid_dim, self.act_dim)]
        return nn.Sequential(*model)
    
    def get_action_dist(self, model_state):
        mean = self.model(model_state)
        std = torch.exp(self.log_std)
        if self.dist == 'normal':
            return Normal(mean, std)
        
    def forward(self, model_state):
        action_dist = self.get_action_dist(model_state)
        action = action_dist.rsample()  # Reparameterization trick
        return action, action_dist