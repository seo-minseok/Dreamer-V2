import torch.nn as nn
from torch.distributions import Bernoulli, Independent, Normal

class DenseModel(nn.Module):
    def __init__(self, in_dim, out_dim, info):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = info['num_layers']
        self.hid_dim = info['hid_dim']
        self.activation = info['activation']
        self.dist = info['dist']
        
        self.model = self._build_model()

    def _build_model(self):
        model = [nn.Linear(self.in_dim, self.hid_dim)]
        model += [self.activation()]

        for i in range(self.num_layers - 1):
            model += [nn.Linear(self.hid_dim, self.hid_dim)]
            model += [self.activation()]

        model += [nn.Linear(self.hid_dim, self.out_dim)]
        return nn.Sequential(*model)
    
    def forward(self, x):
        x = self.model(x)

        if self.dist == 'normal':
            return Independent(Normal(x, 1), self.out_dim)
        elif self.dist == 'binary':
            return Independent(Bernoulli(logits=x), self.out_dim)  # x가 logit임을 명시하지 않으면 probs인지 logits인지 몰라서 에러 발생
        elif self.dist == None:
            return x
        else:
            raise NotImplementedError