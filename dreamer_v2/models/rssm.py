import torch
import torch.nn as nn

from dreamer_v2.utils.rssm import RSSMUtils, RSSMDiscState, RSSMContState

class RSSM(nn.Module, RSSMUtils):
    def __init__(self, act_dim, embed_dim, rssm_type, info, device):
        nn.Module.__init__(self)
        RSSMUtils.__init__(self, rssm_type, info)

        self.act_dim = act_dim
        self.embed_dim = embed_dim
        self.hid_dim = info['hid_dim']
        self.activation = info['activation']
        self.device = device

        self.rnn = nn.GRUCell(self.deter_size, self.deter_size)
        self.fc_embed_state_action = self._build_embed_state_action()
        self.fc_prior = self._build_temporal_prior()
        self.fc_posterior = self._build_temporal_posterior()

    def _build_embed_state_action(self):
        '''
        stochastic state (previous) + action (previous) -> deterministic state (current)
        s_{t-1}, a_{t-1} -> h_t
        '''
        fc_embed_state_action = [nn.Linear(self.stoch_size + self.act_dim, self.deter_size)]
        fc_embed_state_action += [self.activation()]
        return nn.Sequential(*fc_embed_state_action)
    
    def _build_temporal_prior(self):
        '''
        deterministic state (previous) + deterministic state (current) -> stochastic state (current)
        h_{t-1}, h_t -> s_t
        '''
        temporal_prior = [nn.Linear(self.deter_size, self.hid_dim)]
        temporal_prior += [self.activation()]
        if self.rssm_type == 'discrete':
            temporal_prior += [nn.Linear(self.hid_dim, self.stoch_size)]
        elif self.rssm_type == 'continuous':
            temporal_prior += [nn.Linear(self.hid_dim, self.stoch_size * 2)]
        return nn.Sequential(*temporal_prior)
    
    def _build_temporal_posterior(self):
        '''
        deterministic state (current) + observation embedding (current) -> stochastic state (current)
        h_t + o_t -> s_t
        '''
        temporal_posterior = [nn.Linear(self.deter_size + self.embed_dim, self.hid_dim)]
        temporal_posterior += [self.activation()]
        if self.rssm_type == 'discrete':
            temporal_posterior += [nn.Linear(self.hid_dim, self.stoch_size)]
        elif self.rssm_type == 'continuous':
            temporal_posterior += [nn.Linear(self.hid_dim, self.stoch_size * 2)]
        return nn.Sequential(*temporal_posterior)
    
    def rssm_imagine(self, prev_action, prev_rssm_state, nonterms=True):
        '''
        Compute prior state (1-step)
        1. fc_embed_state_action: stochastic state (previous) + action (previous) -> deterministic state (current)
        2. rnn: deterministic state (current) + deterministic state (previous) -> deterministic state (current)
        3. fc_prior: deterministic state (current) -> stochastic state (current)
        '''
        state_action_embed = self.fc_embed_state_action(torch.cat([prev_rssm_state.stoch * nonterms, prev_action], dim=-1))  # 1. s_{t-1}, a_{t-1} -> h_t
        deter_state = self.rnn(state_action_embed, prev_rssm_state.deter * nonterms)                                         # 2. f(h_{t-1}, s_{t-1}, a_{t-1}) -> h_t
        if self.rssm_type == 'discrete':
            prior_logit = self.fc_prior(deter_state)                                                                         # 3. s_t ~ p(s_t | h_t)
            stats = {'logit':prior_logit}
            prior_stoch_state = self.get_stoch_state(stats)
            prior_rssm_state = RSSMDiscState(prior_logit, prior_stoch_state, deter_state)
        elif self.rssm_type == 'continuous':
            prior_mean, prior_std = torch.chunk(self.fc_prior(deter_state), 2, dim=-1)
            stats = {'mean':prior_mean, 'std':prior_std}
            prior_stoch_state, std = self.get_stoch_state(stats)
            prior_rssm_state = RSSMContState(prior_mean, std, prior_stoch_state, deter_state)
        return prior_rssm_state
    
    def rollout_imagination(self, horizon, actor, prev_rssm_state):
        '''
        Compute prior states (horizon-step)
        1. for t in range(horizon)
            1-1. compute prior state (1-step)
        2. stack prior states
        '''
        rssm_state = prev_rssm_state  # (batch, posterior)
        next_rssm_states = []
        imag_log_probs = []
        action_entropy = []
        for t in range(horizon):
            action, action_dist = actor((self.get_model_state(rssm_state)).detach())
            rssm_state = self.rssm_imagine(action, rssm_state)                           # prior: s_t ~ p(s_t | h_t)
            next_rssm_states.append(rssm_state)
            # imag_log_probs.append(action_dist.log_prob(torch.round(action.detach())))  # TODO: discrete action이라고 가정했기 때문에 torch.round로 감쌈 -> continuous action이므로 수정 필요
            imag_log_probs.append(action_dist.log_prob(action.detach()))
            action_entropy.append(action_dist.entropy())
        
        next_rssm_states = self.rssm_stack_states(next_rssm_states, dim=0)
        imag_log_probs = torch.stack(imag_log_probs, dim=0)
        action_entropy = torch.stack(action_entropy, dim=0)
        return next_rssm_states, imag_log_probs, action_entropy
    
    def rssm_observe(self, obs_embed, prev_action, prev_nonterm, prev_rssm_state):
        '''
        Compute prior/posterior state (1-step)
        '''
        prior_rssm_state = self.rssm_imagine(prev_action, prev_rssm_state, prev_nonterm)  # 이 코드가 느릴 수 밖에 없는 이유가 posterior states 계산 시에는 deterministic state만 필요한데, deterministic state만 따로 구하지 않고, RSSM 네트워크 연산을 통해 stochastic state까지 다 구함
        deter_state = prior_rssm_state.deter
        x = torch.cat([deter_state, obs_embed], dim=-1)
        if self.rssm_type == 'discrete':
            posterior_logit = self.fc_posterior(x)  # encoder: s_t ~ q(s_t | h_t, o_t)
            stats = {'logit':posterior_logit}
            posterior_stoch_state = self.get_stoch_state(stats)
            posterior_rssm_state = RSSMDiscState(posterior_logit, posterior_stoch_state, deter_state)
        elif self.rssm_type == 'continuous':
            posterior_mean, posterior_std = torch.chunk(self.fc_posterior(x), 2, dim=-1)
            stats = {'mean':posterior_mean, 'std':posterior_std}
            posterior_stoch_state, std = self.get_stoch_state(stats)
            posterior_rssm_state = RSSMContState(posterior_mean, std, posterior_stoch_state, deter_state)
        return prior_rssm_state, posterior_rssm_state
    
    def rollout_observation(self, seq_len, obs_embed, actions, nonterms, prev_rssm_state):
        '''
        Compute prior/posterior states (seq_len-step)
        1. for t in range(seq_len)
            1-1. compute posterior state (1-step)
        2. stack posterior states
        '''
        priors = []
        posteriors = []
        for t in range(seq_len):
            prev_action = actions[t] * nonterms[t]
            prior_rssm_state, posterior_rssm_state = self.rssm_observe(obs_embed[t], prev_action, nonterms[t], prev_rssm_state)  # prev_rssm_state (B, RSSMState): 처음에만 initial state 사용, 이후에는 이전 시점의 posterior 사용
            priors.append(prior_rssm_state)
            posteriors.append(posterior_rssm_state)
            prev_rssm_state = posterior_rssm_state
        prior = self.rssm_stack_states(priors, dim=0)
        posterior = self.rssm_stack_states(posteriors, dim=0)  # (L, B, RSSMState)
        return prior, posterior