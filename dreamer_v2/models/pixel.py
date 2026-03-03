import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)

def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)

class ObsEncoder(nn.Module):
    def __init__(self, input_shape, embed_dim, info):
        super().__init__()

        self.input_shape = input_shape
        self.embed_dim = embed_dim    # embed_dim: 이때는 실제로 embedding 되는 사이즈를 의미
        self.depth = info['depth']    # 32
        self.kernel = info['kernel']  # 4
        self.stride = info['stride']  # 2
        self.activation = info['activation']
        self.convolutions = self._build_convolutions()
        self.fc_embed = nn.Identity() if self.embed_size == self.embed_dim else nn.Linear(self.embed_size, self.embed_dim)

    def _build_convolutions(self):
        convolutions = [
            nn.Conv2d(self.input_shape[0], self.depth, self.kernel, self.stride),
            self.activation(),
            nn.Conv2d(self.depth, self.depth * 2, self.kernel, self.stride),
            self.activation(),
            nn.Conv2d(self.depth * 2, self.depth * 4, self.kernel, self.stride),
            self.activation(),
            nn.Conv2d(self.depth * 4, self.depth * 8, self.kernel, self.stride),
            self.activation(),
        ]
        return nn.Sequential(*convolutions)
    
    def forward(self, obs):
        batch_shape = obs.shape[:-3]
        img_shape = obs.shape[-3:]
        embed = self.convolutions(obs.reshape(-1, *img_shape))
        embed = torch.reshape(embed, (*batch_shape, -1))
        embed = self.fc_embed(embed)
        return embed

    @property
    def embed_size(self):
        conv1_shape = conv_out_shape(self.input_shape[1:], 0, self.kernel, self.stride)
        conv2_shape = conv_out_shape(conv1_shape, 0, self.kernel, self.stride)
        conv3_shape = conv_out_shape(conv2_shape, 0, self.kernel, self.stride)
        conv4_shape = conv_out_shape(conv3_shape, 0, self.kernel, self.stride)
        embed_size = int(self.depth * 8 * np.prod(conv4_shape).item())
        return embed_size


class ObsDecoder(nn.Module):
    def __init__(self, output_shape, embed_dim, info):
        super().__init__()

        self.output_shape = output_shape  # output_shape = obs_shape
        self.embed_dim = embed_dim        # embed_dim: embed_dim이라고 변수명을 지엇지만 실제로는 deter state + stoch state이다.
        self.depth = info['depth']        # 32
        self.kernel = info['kernel']      # 5
        self.stride = info['stride']      # 2
        self.activation = info['activation']
        self.fc_embed = nn.Identity() if self.embed_dim == self.embed_size else nn.Linear(self.embed_dim, self.embed_size)
        self.deconvolutions = self._build_deconvolutions()

    def _build_deconvolutions(self):
        deconvolutions = [
            nn.ConvTranspose2d(self.embed_size, self.depth * 4, self.kernel, self.stride),
            self.activation(),
            nn.ConvTranspose2d(self.depth * 4, self.depth * 2, self.kernel, self.stride),
            self.activation(),
            nn.ConvTranspose2d(self.depth * 2, self.depth, self.kernel + 1, self.stride),
            self.activation(),
            nn.ConvTranspose2d(self.depth, self.output_shape[0], self.kernel + 1, self.stride)
        ]
        return nn.Sequential(*deconvolutions)
    
    def forward(self, x):
        batch_shape = x.shape[:-1]
        embed_size = x.shape[-1]
        squeezed_size = np.prod(batch_shape).item()
        
        x = x.reshape(squeezed_size, embed_size)
        x = self.fc_embed(x)
        x = torch.reshape(x, (squeezed_size, *self.conv_shape))
        x = self.deconvolutions(x)
        mean = torch.reshape(x, (*batch_shape, *self.output_shape))
        obs_dist = Independent(Normal(mean, 1), len(self.output_shape))
        return obs_dist

    @property
    def embed_size(self):
        conv1_shape = conv_out_shape(self.output_shape[1:], 0, self.kernel, self.stride)
        conv2_shape = conv_out_shape(conv1_shape, 0, self.kernel, self.stride)
        conv3_shape = conv_out_shape(conv2_shape, 0, self.kernel, self.stride)
        conv4_shape = conv_out_shape(conv3_shape, 0, self.kernel, self.stride)
        self.conv_shape = (self.depth * 8, *conv4_shape)
        embed_size = int(np.prod(self.conv_shape).item())
        return embed_size
        # embed_size = int(self.depth * 8 * np.prod(conv4_shape).item())
        # return embed_size