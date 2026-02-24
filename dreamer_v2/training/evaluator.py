import numpy as np

import torch
from tqdm import tqdm

from dreamer_v2.models import ContinuousActor, ObsEncoder, ObsDecoder, RSSM

class Evaluator(object):
    def __init__(self, config, device):
        self.config = config
        self.device = device

        self._init_models()

    def _init_models(self):
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

        # World Model
        self.RSSM = RSSM(act_dim=act_size, embed_dim=embed_dim, rssm_type=self.config.rssm_type, info=self.config.rssm_info, device=self.device).to(self.device)

        # Actor-Critic
        self.actor = ContinuousActor(deter_size=deter_size, stoch_size=stoch_size, act_dim=act_size, info=self.config.actor_info).to(self.device)

        # Encoder / Decoder
        self.encoder = ObsEncoder(input_shape=obs_shape, embed_dim=embed_dim, info=self.config.encoder_info).to(self.device)
        self.decoder = ObsDecoder(output_shape=obs_shape, embed_dim=model_state_size, info=self.config.decoder_info).to(self.device)

    def load_model(self, model_path):
        saved_dict = torch.load(model_path)
        self.RSSM.load_state_dict(saved_dict["RSSM"])
        self.encoder.load_state_dict(saved_dict["encoder"])
        self.decoder.load_state_dict(saved_dict["decoder"])
        self.actor.load_state_dict(saved_dict["actor"])

    def eval_saved_agent(self, env, model_path):
        self.load_model(model_path)
        eval_episodes = self.config.eval_episode

        total_rewards, total_costs = [], []

        eval_pbar = tqdm(total=eval_episodes, desc="  ┗ [Eval] Progress", position=1, leave=False)

        for _ in range(eval_episodes):
            obs, done = env.reset(), False
            episode_reward, episode_cost = 0, 0

            prev_rssm_state = self.RSSM.init_rssm_state(1)
            prev_action = torch.zeros(1, self.config.act_size).to(self.device)

            while not done:
                with torch.no_grad():
                    embed = self.encoder(obs).to(self.device)
                    _, posterior_rssm_state = self.RSSM.rssm_observe(embed, prev_action, not done, prev_rssm_state)
                    model_state = self.RSSM.get_model_state(posterior_rssm_state)
                    
                    action, _ = self.actor(model_state)
                    
                next_obs, reward, cost, done = env.step(action[0].cpu().numpy())    

                prev_rssm_state = posterior_rssm_state
                prev_action = action
                obs = next_obs

                episode_reward += reward
                episode_cost += cost

            total_rewards.append(episode_reward)
            total_costs.append(episode_cost)

            eval_pbar.update(1)
        eval_pbar.close()

        return np.mean(total_rewards), np.mean(total_costs)