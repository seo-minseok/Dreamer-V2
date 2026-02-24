import numpy as np

import torch

from dreamer_v2.envs.safety_gym import preprocess_observation, postprocess_observation

class Buffer:
    def __init__(self, size, obs_shape, act_size, bit_depth, device):
        self.size = size
        self.obs_shape = obs_shape
        self.act_size = act_size
        self.bit_depth = bit_depth
        self.device = device
        self.full = False
        self.idx = 0

        self.observations = np.empty((size, *obs_shape), dtype=np.float32)
        self.actions = np.empty((size, act_size), dtype=np.float32)
        self.rewards = np.empty((size, ), dtype=np.float32)
        self.costs = np.empty((size, ), dtype=np.float32)
        self.dones = np.empty((size, ), dtype=np.float32)

    def add(self, obs, act, reward, cost, done):
        self.observations[self.idx] = postprocess_observation(obs, self.bit_depth)
        self.actions[self.idx] = act
        self.rewards[self.idx] = reward
        self.costs[self.idx] = cost
        self.dones[self.idx] = done
        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0

    def _sample_indices(self, L):
        valid_idx = False
        while not valid_idx:
            idx = np.random.randint(0, self.size if self.full else self.idx - L)
            idxs = np.arange(idx, idx + L) % self.size
            valid_idx = not self.idx in idxs[1:]
        return idxs
    
    def _retrieve_batch(self, idxs, n, L):
        vec_idxs = idxs.transpose().reshape(-1)                                         # vec_idxs: (n * L, )
        observations = torch.as_tensor(self.observations[vec_idxs].astype(np.float32))  # observations: (n * L, C, H, W)
        preprocess_observation(observations, self.bit_depth)
        return observations.reshape(L, n, *self.obs_shape), self.actions[vec_idxs].reshape(L, n, -1), self.rewards[vec_idxs].reshape(L, n), self.costs[vec_idxs].reshape(L, n), self.dones[vec_idxs].reshape(L, n)
    
    def sample(self, n, L):
        L = L + 1
        obs, acts, rewards, costs, dones = self._retrieve_batch(np.asarray([self._sample_indices(L) for _ in range(n)]), n, L)  # [self._sample_indices(L) for _ in range(n)]: (n, L)
        obs, acts, rewards, costs, dones = self._shif_sequences(obs, acts, rewards, costs, dones)

        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        acts = torch.tensor(acts, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(-1)
        costs = torch.tensor(costs, dtype=torch.float32).to(self.device).unsqueeze(-1)
        non_terms = torch.tensor(1 - dones, dtype=torch.float32).to(self.device).unsqueeze(-1)

        return obs, acts, rewards, costs, non_terms
        

    def _shif_sequences(self, obs, acts, rewards, costs, dones):
        '''
        실제 deploy 시에는 initial state를 입력으로 주어 rollout 하는게 맞지만, RSSM 학습 시에는 데이터 간의 인과관계를 학습하기 위해 obs[1:]부터 입력으로 넣음
        '''
        obs = obs[1:]           # t + 1 to seq_len
        acts = acts[:-1]        # t to seq_len - 1
        rewards = rewards[:-1]  # t to seq_len - 1
        costs = costs[:-1]      # t to seq_len - 1
        dones = dones[:-1]      # t to seq_len - 1
        return obs, acts, rewards, costs, dones