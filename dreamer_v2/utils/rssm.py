from collections import namedtuple
from typing import Union

import torch
import torch.nn.functional as F
from torch.distributions import Independent, Normal, OneHotCategorical

RSSMDiscState = namedtuple('RSSMDiscState', ['logit', 'stoch', 'deter'])
RSSMContState = namedtuple('RSSMContState', ['mean', 'std', 'stoch', 'deter'])
RSSMState = Union[RSSMDiscState, RSSMContState]

def seq_to_batch(sequence_data, batch_size, seq_len):
    # (L, B, ) -> (L * B, )
    shape = tuple(sequence_data.shape)
    batch_data = torch.reshape(sequence_data, [shape[0] * shape[1], *shape[2:]])
    return batch_data

def batch_to_seq(batch_data, batch_size, seq_len):
    # (L * B, ) -> (L, B, )
    shape = tuple(batch_data.shape)
    seq_data = torch.reshape(batch_data, [seq_len, batch_size, *shape[1:]])
    return seq_data

class RSSMUtils:
    def __init__(self, rssm_type, info):
        self.rssm_type = rssm_type
        if self.rssm_type == 'discrete':
            self.deter_size = info['deter_size']
            self.class_size = info['class_size']
            self.category_size = info['category_size']
            self.stoch_size = self.class_size * self.category_size
        elif self.rssm_type == 'continuous':
            self.deter_size = info['deter_size']
            self.stoch_size = info['stoch_size']
            self.min_std = info['min_std']

    def init_rssm_state(self, batch_size, **kwargs):
        # 실질적으로 쓰는 값은 deter, stoch 밖에 안 씀 -> logit, mean, std는 모두 stoch_size이기 때문에 stoch_size로 생성 (_build_temporal_prior()의 temporal_prior 모듈을 보면 알 수 있음)
        if self.rssm_type == 'discrete':
            return RSSMDiscState(
                torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),  # logit
                torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),  # stoch
                torch.zeros(batch_size, self.deter_size, **kwargs).to(self.device)   # deter
            )
        elif self.rssm_type == 'continuous':
            return RSSMContState(
                torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),  # mean
                torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),  # std
                torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),  # stoch
                torch.zeros(batch_size, self.deter_size, **kwargs).to(self.device)   # deter
            )
        
    def rssm_seq_to_batch(self, rssm_state, batch_size, seq_len):
        if self.rssm_type == 'discrete':
            return RSSMDiscState(
                seq_to_batch(rssm_state.logit[:seq_len], batch_size, seq_len),
                seq_to_batch(rssm_state.stoch[:seq_len], batch_size, seq_len),
                seq_to_batch(rssm_state.deter[:seq_len], batch_size, seq_len)
            )
        elif self.rssm_type == 'continuous':
            return RSSMContState(
                seq_to_batch(rssm_state.mean[:seq_len], batch_size, seq_len),
                seq_to_batch(rssm_state.std[:seq_len], batch_size, seq_len),
                seq_to_batch(rssm_state.stoch[:seq_len], batch_size, seq_len),
                seq_to_batch(rssm_state.deter[:seq_len], batch_size, seq_len)
            )
        
    def rssm_batch_to_seq(self, rssm_state, batch_size, seq_len):
        if self.rssm_type == 'discrete':
            return RSSMDiscState(
                batch_to_seq(rssm_state.logit, batch_size, seq_len),
                batch_to_seq(rssm_state.stoch, batch_size, seq_len),
                batch_to_seq(rssm_state.deter, batch_size, seq_len)
            )
        elif self.rssm_type == 'continuous':
            return RSSMContState(
                batch_to_seq(rssm_state.mean, batch_size, seq_len),
                batch_to_seq(rssm_state.std, batch_size, seq_len),
                batch_to_seq(rssm_state.stoch, batch_size, seq_len),
                batch_to_seq(rssm_state.deter, batch_size, seq_len)
            )
         
    def get_dist(self, rssm_state):
        if self.rssm_type == 'discrete':
            '''
            - OneHotCategorical: (batch, category, class) -> (batch, category)
            - Independent: (batch, category) -> (batch, )
            '''
            shape = rssm_state.logit.shape
            logit = torch.reshape(rssm_state.logit, shape=(*shape[:-1], self.category_size, self.class_size))  # logit은 self.stoch_size(class_size * category_size) 크기, 이를 다시 class_size, category_size로 reshape
            return Independent(OneHotCategorical(logits=logit), 1)
        elif self.rssm_type == 'continuous':
            '''
            - Normal: (batch, stoch_size) -> (batch, stoch_size)
            - Independent: (batch, stoch_size) -> (batch, )
            '''
            return Independent(Normal(rssm_state.mean, rssm_state.std), 1)
        
    def get_stoch_state(self, stats):
        if self.rssm_type == 'discrete':
            logit = stats['logit']
            shape = logit.shape
            logit = torch.reshape(logit, shape=(*shape[:-1], self.category_size, self.class_size))
            dist = OneHotCategorical(logits=logit)
            stoch = dist.sample()
            stoch += dist.probs - dist.probs.detach()              # Straight-Throguh Gradients with Automatic Differentiation
            return torch.flatten(stoch, start_dim=-2, end_dim=-1)  # (batch, category_size * class_size)
        elif self.rssm_type == 'continuous':
            mean = stats['mean']
            std = stats['std']
            std = F.softplus(std) + self.min_std
            return mean + std * torch.randn_like(mean), std        # Reparameterization trick
        
    def get_model_state(self, rssm_state):
        if self.rssm_type == 'discrete':
            return torch.cat((rssm_state.deter, rssm_state.stoch), dim=-1)
        elif self.rssm_type == 'continuous':
            return torch.cat((rssm_state.deter, rssm_state.stoch), dim=-1)
        
    def rssm_stack_states(self, rssm_states, dim):
        if self.rssm_type == 'discrete':
            return RSSMDiscState(
                torch.stack([state.logit for state in rssm_states], dim=dim),
                torch.stack([state.stoch for state in rssm_states], dim=dim),
                torch.stack([state.deter for state in rssm_states], dim=dim)
            )
        elif self.rssm_type == 'continuous':
            return RSSMContState(
                torch.stack([state.mean for state in rssm_states], dim=dim),
                torch.stack([state.std for state in rssm_states], dim=dim),
                torch.stack([state.stoch for state in rssm_states], dim=dim),
                torch.stack([state.deter for state in rssm_states], dim=dim)
            )
    
    def rssm_detach(self, rssm_state):
        if self.rssm_type == 'discrete':
            return RSSMDiscState(
                rssm_state.logit.detach(),
                rssm_state.stoch.detach(),
                rssm_state.deter.detach()
            )
        elif self.rssm_type == 'continuous':
            return RSSMContState(
                rssm_state.mean.detach(),
                rssm_state.std.detach(),
                rssm_state.stoch.detach(),
                rssm_state.deter.detach()
            )