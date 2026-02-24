from typing import Iterable
from collections import defaultdict

import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import kl_divergence
from torch.nn.utils import clip_grad_norm_

from dreamer_v2.models import ContinuousActor, DenseModel, ObsEncoder, ObsDecoder, RSSM
from dreamer_v2.utils.buffer import Buffer
from dreamer_v2.utils.utils import compute_return


def get_parameters(modules):
    params = []
    for module in modules:
        params += list(module.parameters())
    return params

class FreezeParameters:
    def __init__(self, modules: Iterable[nn.Module]):
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(modules)]
    
    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]


class Trainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.batch_size = config.batch_size
        self.seq_len = config.seq_len
        self.seed_episodes = config.seed_episodes

        self._model_initialize()
        self._optim_initialize()

    def _model_initialize(self):
        obs_shape = self.config.obs_shape
        act_size = self.config.act_size
        deter_size = self.config.rssm_info['deter_size']
        if self.config.rssm_type == 'continuous':
            stoch_size = self.config.rssm_info['stoch_size']
        elif self.config.rssm_type == 'discrete':
            category_size = self.config.rssm_info['category_size']
            class_size = self.config.rssm_info['class_size']
            stoch_size = category_size * class_size
        model_state_size = deter_size + stoch_size
        embed_dim = self.config.encoder_info['embed_dim']

        self.buffer = Buffer(size=self.config.capacity, obs_shape=obs_shape, act_size=act_size, bit_depth=self.config.bit_depth, device=self.device)

        # World Model
        self.RSSM = RSSM(act_dim=act_size, embed_dim=embed_dim, rssm_type=self.config.rssm_type, info=self.config.rssm_info, device=self.device).to(self.device)
        self.reward_model = DenseModel(in_dim=model_state_size, out_dim=1, info=self.config.reward_info).to(self.device)
        self.cost_model = DenseModel(in_dim=model_state_size, out_dim=1, info=self.config.cost_info).to(self.device)
        self.discount_model = DenseModel(in_dim=model_state_size, out_dim=1, info=self.config.discount_info).to(self.device)

        # Actor-Critic
        self.actor = ContinuousActor(deter_size=deter_size, stoch_size=stoch_size, act_dim=act_size, info=self.config.actor_info).to(self.device)
        self.critic = DenseModel(in_dim=model_state_size, out_dim=1, info=self.config.critic_info).to(self.device)
        self.target_critic = DenseModel(in_dim=model_state_size, out_dim=1, info=self.config.critic_info).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Encoder / Decoder
        self.encoder = ObsEncoder(input_shape=obs_shape, embed_dim=embed_dim, info=self.config.encoder_info).to(self.device)
        self.decoder = ObsDecoder(output_shape=obs_shape, embed_dim=model_state_size, info=self.config.decoder_info).to(self.device)

    def _optim_initialize(self):
        world_lr = self.config.lr['world']
        actor_lr = self.config.lr['actor']
        critic_lr = self.config.lr['critic']

        self.world_list = [self.encoder, self.RSSM, self.reward_model, self.cost_model, self.discount_model]
        self.actor_list = [self.actor]
        self.critic_list = [self.critic]

        self.world_optimizer = optim.Adam(get_parameters(self.world_list), lr=world_lr)
        self.actor_optimizer = optim.Adam(get_parameters(self.actor_list), lr=actor_lr)
        self.critic_optimizer = optim.Adam(get_parameters(self.critic_list), lr=critic_lr)

    def update_target(self):
        mix = self.config.slow_target_fraction if self.config.use_slow_target else 1

        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(mix * param.data + (1 - mix) * target_param.data)

    def save_model(self, iter):
        save_dict = self.get_save_dict()
        model_dir = self.config.model_dir
        save_path = os.path.join(model_dir, 'models_%d.pth' % iter)
        torch.save(save_dict, save_path)

    def get_save_dict(self):
        return {
            'RSSM': self.RSSM.state_dict(),
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'reward_model': self.reward_model.state_dict(),
            'cost_model': self.cost_model.state_dict(),
            'discount_model': self.discount_model.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }
    
    def collect_seed_episodes(self, env):
        for _ in range(self.seed_episodes):
            obs, done = env.reset(), False
            while not done:
                action = env.sample_random_action()
                next_obs, reward, cost, done = env.step(action)
                self.buffer.add(obs, action, reward, cost, done)
                obs = next_obs

    def train_batch(self, train_metrics):
        train_stats = defaultdict(list)

        for _ in range(self.config.collect_intervals):
            obs, acts, rewards, costs, nonterms = self.buffer.sample(self.batch_size, self.seq_len)

            world_loss, kl_loss, obs_loss, reward_loss, cost_loss, discount_loss, prior_dist ,post_dist, posterior = self.representation_loss(obs, acts, rewards, costs, nonterms)

            self.world_optimizer.zero_grad()
            world_loss.backward()
            self.world_optimizer.step()

            actor_loss, critic_loss = self.actor_critic_loss(posterior)

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            grad_norm_actor = clip_grad_norm_(get_parameters(self.actor_list), self.config.grad_clip)
            grad_norm_critic = clip_grad_norm_(get_parameters(self.critic_list), self.config.grad_clip)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            with torch.no_grad():
                prior_entropy = torch.mean(prior_dist.entropy())
                post_entropy = torch.mean(post_dist.entropy())

            train_stats['world_loss'].append(world_loss.item())
            train_stats['obs_loss'].append(obs_loss.item())
            train_stats['reward_loss'].append(reward_loss.item())
            train_stats['cost_loss'].append(cost_loss.item())
            train_stats['discount_loss'].append(discount_loss.item())
            train_stats['prior_entropy'].append(prior_entropy.item())
            train_stats['post_entropy'].append(post_entropy.item())
            train_stats['kl_loss'].append(kl_loss.item())
            train_stats['actor_loss'].append(actor_loss.item())
            train_stats['critic_loss'].append(critic_loss.item())

        for key, values in train_stats.items():
            train_metrics[key] = np.mean(values)

        return train_metrics

    def representation_loss(self, obs, acts, rewards, costs, nonterms):
        embed = self.encoder(obs)
        prev_rssm_state = self.RSSM.init_rssm_state(self.batch_size)
        prior, posterior = self.RSSM.rollout_observation(seq_len=self.seq_len, obs_embed=embed, actions=acts, nonterms=nonterms, prev_rssm_state=prev_rssm_state)

        post_model_state = self.RSSM.get_model_state(posterior)           # post_model_state: (L, B, deter + stoch)

        obs_dist = self.decoder(post_model_state[:-1])
        reward_dist = self.reward_model(post_model_state[:-1])
        cost_dist = self.cost_model(post_model_state[:-1])
        discount_dist = self.discount_model(post_model_state[:-1])

        obs_loss = self._obs_loss(obs_dist, obs[:-1])                     # using current latent state to predict current observation
        reward_loss = self._reward_loss(reward_dist, rewards[1:])         # using current latent state to predict next reward
        cost_loss = self._cost_loss(cost_dist, costs[1:])                 # using current latent state to predict next cost
        discount_loss = self._discount_loss(discount_dist, nonterms[1:])  # using current latent state to predict next discount
        prior_dist, post_dist, kl_loss = self._kl_loss(prior, posterior)

        world_loss = obs_loss + reward_loss + cost_loss + self.config.loss_scale['discount'] * discount_loss + self.config.loss_scale['kl'] * kl_loss

        return world_loss, kl_loss, obs_loss, reward_loss, cost_loss, discount_loss, prior_dist, post_dist, posterior
    
    def _obs_loss(self, obs_dist, obs):
        obs_loss = -torch.mean(obs_dist.log_prob(obs))
        return obs_loss
    
    def _reward_loss(self, reward_dist, rewards):
        reward_loss = -torch.mean(reward_dist.log_prob(rewards))
        return reward_loss
    
    def _cost_loss(self, cost_dist, costs):
        cost_loss = -torch.mean(cost_dist.log_prob(costs))
        return cost_loss
    
    def _discount_loss(self, discount_dist, nonterms):
        discount_target = nonterms.float()
        discount_loss = -torch.mean(discount_dist.log_prob(discount_target))
        return discount_loss
    
    def _kl_loss(self, prior, posterior):
        prior_dist = self.RSSM.get_dist(prior)
        post_dist = self.RSSM.get_dist(posterior)
        
        if self.config.kl_info['use_kl_balance']:
            alpha = self.config.kl_info['kl_balance_scale']
            kl_lhs = torch.mean(kl_divergence(self.RSSM.get_dist(self.RSSM.rssm_detach(posterior)), prior_dist))
            kl_rhs = torch.mean(kl_divergence(post_dist, self.RSSM.get_dist(self.RSSM.rssm_detach(prior))))
            if self.config.kl_info['use_free_nats']:
                free_nats = self.config.kl_info['free_nats']
                kl_lhs = torch.max(kl_lhs,kl_lhs.new_full(kl_lhs.size(), free_nats))  # new_full: 텐서와 동일한 디바이스와 데이터 타입을 가지는 텐서를 생성하고, 값을 free_nats로 채움
                kl_rhs = torch.max(kl_rhs,kl_rhs.new_full(kl_rhs.size(), free_nats))
            kl_loss = alpha * kl_lhs + (1 - alpha) * kl_rhs
        else:
            kl_loss = torch.mean(kl_divergence(post_dist, prior_dist))
            if self.config.kl_info['use_free_nats']:
                free_nats = self.config.kl_info['free_nats']
                kl_loss = torch.max(kl_loss, kl_loss.new_full(kl_loss.size(), free_nats))
        
        return prior_dist, post_dist, kl_loss
    
    def actor_critic_loss(self, posterior):
        with torch.no_grad():
            batched_posterior = self.RSSM.rssm_detach(self.RSSM.rssm_seq_to_batch(posterior, self.batch_size, self.seq_len - 1))

        with FreezeParameters(self.world_list):
            imag_rssm_states, imag_log_prob, policy_entropy = self.RSSM.rollout_imagination(self.config.horizon, self.actor, batched_posterior)

        imag_model_states = self.RSSM.get_model_state(imag_rssm_states)

        with FreezeParameters(self.world_list + self.critic_list + [self.target_critic] + [self.discount_model]):
            imag_reward_dist = self.reward_model(imag_model_states)
            imag_reward = imag_reward_dist.mean
            
            imag_value_dist = self.target_critic(imag_model_states)
            imag_value = imag_value_dist.mean

            discount_dist = self.discount_model(imag_model_states)  # Bernoulli distribution
            discount_arr = self.config.discount * torch.round(discount_dist.base_dist.probs)

        actor_loss, discount, lambda_returns = self._actor_loss(imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy)
        critic_loss = self._critic_loss(imag_model_states, discount, lambda_returns)
    
        return actor_loss, critic_loss
    
    def _actor_loss(self, imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy):
        lambda_returns = compute_return(imag_reward[:-1], imag_value[:-1], discount_arr[:-1], bootstrap=imag_value[-1], lambda_=self.config.lambda_)

        advantage = (lambda_returns - imag_value[:-1]).detach()
        objective = imag_log_prob[1:].sum(dim=-1).unsqueeze(-1) * advantage

        discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]])
        discount = torch.cumprod(discount_arr[:-1], 0)
        policy_entropy = policy_entropy[1:].sum(dim=-1).unsqueeze(-1)
        actor_loss = -torch.sum(torch.mean(discount * (objective + self.config.actor_entropy_scale * policy_entropy), dim=1))

        return actor_loss, discount, lambda_returns
    
    def _critic_loss(self, imag_model_states, discount, lambda_returns):
        with torch.no_grad():
            value_model_states = imag_model_states[:-1].detach()
            value_discount = discount.detach()
            value_target = lambda_returns.detach()

        value_dist = self.critic(value_model_states)
        critic_loss = -torch.mean(value_discount * value_dist.log_prob(value_target).unsqueeze(-1))

        return critic_loss