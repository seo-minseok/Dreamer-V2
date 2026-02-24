import torch.nn as nn
from dataclasses import dataclass, field

@dataclass
class SafetyGymConfig:
    # env
    obs_shape: tuple
    act_size: int
    bit_depth: int = 5

    horizon: int = 15
    discount: float = 0.99
    lambda_: float =0.95

    # training
    seed_episodes: int = 5
    train_steps: int = int(2e6)
    train_every: int = 50
    collect_intervals: int = 5
    batch_size: int = 50
    seq_len: int = 50
    eval_episode: int = 5
    save_every: int = int(1e4)
    model_dir: str = 'results'

    # buffer
    capacity: int = int(1e6)

    # rssm
    rssm_type: str = 'discrete'  # 'continuous' or 'discrete'
    rssm_info: dict = field(default_factory=lambda:{'deter_size': 200, 'hid_dim': 200, 'stoch_size': 20, 'class_size': 32, 'category_size': 32, 'activation':nn.ELU ,'min_std': 0.1})

    # actor
    actor_info: dict = field(default_factory=lambda:{'num_layers': 3, 'hid_dim': 100,  'dist': 'normal', 'activation':nn.ELU})
    actor_entropy_scale: float = 1e-3

    # critic
    critic_info: dict = field(default_factory=lambda:{'num_layers': 3, 'hid_dim': 100, 'dist': 'normal', 'activation': nn.ELU})

    # reward / cost model
    reward_info: dict = field(default_factory=lambda:{'num_layers': 3, 'hid_dim': 100, 'dist': 'normal', 'activation': nn.ELU})
    cost_info: dict = field(default_factory=lambda:{'num_layers': 3, 'hid_dim': 100, 'dist': 'normal', 'activation': nn.ELU})

    # discount model
    discount_info: dict = field(default_factory=lambda:{'num_layers': 3, 'hid_dim': 100, 'dist': 'binary', 'activation': nn.ELU})

    # encoder / decoder
    encoder_info: dict = field(default_factory=lambda:{'num_layers': 3, 'hid_dim': 100, 'embed_dim': 200, 'dist': None, 'activation':nn.ELU, 'depth': 32, 'kernel': 4, 'stride': 2})
    decoder_info: dict = field(default_factory=lambda:{'num_layers': 3, 'hid_dim': 100, 'dist': 'normal', 'activation':nn.ELU, 'depth': 32, 'kernel': 5, 'stride': 2})

    # learning rate
    lr: dict = field(default_factory=lambda:{'world': 2e-4, 'actor': 4e-5, 'critic': 1e-4})

    # objective
    grad_clip: float = 100.0
    loss_scale: dict = field(default_factory=lambda:{'kl': 0.1, 'discount': 5.0})
    use_slow_target: bool = True
    slow_target_update: int = 100
    slow_target_fraction: float = 1.0

    # kl
    kl_info: dict = field(default_factory=lambda:{'use_kl_balance': True, 'kl_balance_scale': 0.8, 'use_free_nats': False, 'free_nats': 0.0})