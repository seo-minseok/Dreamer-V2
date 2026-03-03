import os
os.environ['MUJOCO_GL'] = 'glfw'

from dataclasses import asdict

import csv
import yaml

import time
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from dreamer_v2.envs.safety_gym import SafetyGymEnv
from dreamer_v2.training.config import SafetyGymConfig
from dreamer_v2.training.trainer import Trainer
from dreamer_v2.training.evaluator import Evaluator

def main():
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("🚀 Training on device: ", device)

    start_time = time.time()

    env_id = "SafetyPointGoal1-v0"
    env = SafetyGymEnv(env_id, size=(64, 64), action_repeat=5, bit_depth=5, device=device)
    test_env = SafetyGymEnv(env_id, size=(64, 64), action_repeat=5, bit_depth=5, device=device)

    result_dir = os.path.join('results', datetime.now().strftime('%Y-%m-%d-%H-%M-') + f'{env_id}')
    model_dir = os.path.join(result_dir, 'models')
    model_path = os.path.join(model_dir, 'model.pth')
    os.makedirs(model_dir, exist_ok=True)

    # csv
    csv_path = os.path.join(result_dir, 'result.csv')
    csv_headers = [
        'iter', 'episode', 'EpRet', 'EpCost', 'EvalEpRet', 'EvalEpCost',
        'model_loss', 'obs_loss', 'reward_loss', 'cost_loss', 'discount_loss',
        'prior_entropy', 'post_entropy', 'kl_loss', 'actor_loss', 'critic_loss'
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()

    # configuration
    config = SafetyGymConfig(
        obs_shape=env.observation_space,
        act_size=env.action_space,
        model_dir=model_dir
    )

    config_path = os.path.join(result_dir, 'config.yaml')
    config_dict = asdict(config)

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    

    trainer = Trainer(config, device)
    evaluator = Evaluator(config, device)

    train_metrics = {}
    
    print("🌱 Collecting seed episodes...")
    trainer.collect_seed_episodes(env)

    obs, done = env.reset(), False                                          # obs: (1, 3, 64, 64)
    prev_rssm_state = trainer.RSSM.init_rssm_state(1)                       # prev_rssm_state.deter: (1, 200), prev_rssm_state.stoch: (1, 400)
    prev_action = torch.zeros(1, config.act_size).to(trainer.device)        # prev_action: (1, 2)

    train_reward, train_cost = 0, 0
    eval_reward, eval_cost = 0, 0
    episode_entropies = []
    ep_count = 1

    #=====================================================================#
    #                               Training                              #
    #=====================================================================#
    
    # Main progressbar
    outer_pbar = tqdm(total=trainer.config.train_steps, desc="[Overall Progress]", position=0)
    
    # Subprogressbar
    collect_pbar = tqdm(desc=f"  ┗ [Collect] Episode {ep_count}", position=1, leave=False)

    for iter in range(1, trainer.config.train_steps + 1):
        outer_pbar.set_description(f"[Step {iter}]")
        # 1. Training
        if iter % trainer.config.train_every == 0:
            train_metrics = trainer.train_batch(train_metrics)
            outer_pbar.set_postfix({
                'model_loss': f"{train_metrics.get('model_loss', 0):.4f}",
                'actor_loss': f"{train_metrics.get('actor_loss', 0):.4f}",
                'critic_loss': f"{train_metrics.get('critic_loss', 0):.4f}"
            })

        if iter % trainer.config.slow_target_update == 0:
            trainer.update_target()
        
        if iter % trainer.config.save_every == 0:
            trainer.save_model(iter)
        
        # 2. Data Collection
        with torch.no_grad():
            embed = trainer.encoder(obs).to(trainer.device)                                                     # embed: (1, 200)
            _, posterior_rssm_state = trainer.RSSM.rssm_observe(embed, prev_action, not done, prev_rssm_state)  # posterior_rssm_state.deter: (1, 200), posterior_rssm_state.stoch: (1, 400)
            model_state = trainer.RSSM.get_model_state(posterior_rssm_state)                                    # model_state: (1, 600)
            
            action, action_dist = trainer.actor(model_state)                                                    # action: (1, 2)
            action_entropy = torch.mean(action_dist.entropy()).item()
            episode_entropies.append(action_entropy)

        next_obs, reward, cost, done = env.step(action[0].cpu().numpy())
        train_reward += reward
        train_cost += cost

        trainer.buffer.add(obs, action[0].cpu().numpy(), reward, cost, done)

        # Subprogressbar update
        collect_pbar.update(1)
        collect_pbar.set_postfix({'reward': f'{train_reward:.2f}', 'cost': f'{train_cost:.2f}'})

        if done:
            train_metrics['train_reward'] = train_reward
            train_metrics['train_cost'] = train_cost
            train_metrics['action_entropy'] = np.mean(episode_entropies)

            save_dict = trainer.get_save_dict()
            torch.save(save_dict, model_path)

            # 3. Evaluation
            if ep_count % trainer.config.eval_every == 0:
                eval_reward, eval_cost = evaluator.eval_saved_agent(test_env, model_path)
                outer_pbar.set_postfix({
                    'eval reward': f'{eval_reward:.2f}',
                    'eval cost': f'{eval_cost:.2f}'
                })

            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writerow({
                    'iter': iter,
                    'episode': ep_count,
                    'EpRet': train_metrics.get('train_reward', 0),
                    'EpCost': train_metrics.get('train_cost', 0),
                    'EvalEpRet': eval_reward,
                    'EvalEpCost': eval_cost,
                    'model_loss': train_metrics.get('model_loss', 0),
                    'obs_loss': train_metrics.get('obs_loss', 0),
                    'reward_loss': train_metrics.get('reward_loss', 0),
                    'cost_loss': train_metrics.get('cost_loss', 0),
                    'discount_loss': train_metrics.get('discount_loss', 0),
                    'prior_entropy': train_metrics.get('prior_entropy', 0),
                    'post_entropy': train_metrics.get('post_entropy', 0),
                    'kl_loss': train_metrics.get('kl_loss', 0),
                    'actor_loss': train_metrics.get('actor_loss', 0),
                    'critic_loss': train_metrics.get('critic_loss', 0)
                })

            obs, done = env.reset(), False
            prev_rssm_state = trainer.RSSM.init_rssm_state(1)
            prev_action = torch.zeros(1, config.act_size).to(trainer.device)

            train_reward, train_cost = 0, 0
            eval_reward, eval_cost = 0, 0
            episode_entropies = []
            ep_count += 1

            collect_pbar.close()
            collect_pbar = tqdm(desc=f"  ┗ [Collect] Episode {ep_count}", position=1, leave=False)

        else:
            obs = next_obs
            prev_rssm_state = posterior_rssm_state
            prev_action = action

        outer_pbar.update(1)

    outer_pbar.close()
    collect_pbar.close()
    env.close()

    end_time = time.time()
    duration = end_time - start_time
    print('Training time: {}h {}m {}s'.format(int(duration // 3600), int((duration % 3600) // 60), int(duration % 60)))

if __name__ == "__main__":
    main()