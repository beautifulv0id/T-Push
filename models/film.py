import torch
import torch.nn as nn
from utils.position_encodings import SinusoidalPosEmb

class FILM(nn.Module):

    def __init__(self, in_channels, cond_channels, hidden_channels=512, diffusion_step_embed_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        dsed = diffusion_step_embed_dim
        self.hidden_input_encoder = nn.ModuleList([
            nn.Linear(in_channels, hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
        ]
        )
        self.hidden_cond_encoder = nn.ModuleList([
            nn.Linear(cond_channels + dsed, 2*hidden_channels),
            nn.Linear(cond_channels + dsed, 2*hidden_channels),
            nn.Linear(cond_channels + dsed, 2*hidden_channels),
        ]
        )

        self.out_proj = nn.Linear(hidden_channels, in_channels)
        self.act = nn.ReLU()
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )



    def forward(self, sample, timestep, global_cond):
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        global_feature = torch.cat([
            global_feature, global_cond
        ], axis=-1)

        out = sample
        for input_encoder, cond_encoder in zip(self.hidden_input_encoder, self.hidden_cond_encoder):
            out = self.act(input_encoder(out))
            scale, bias = self.act(cond_encoder(global_feature)).chunk(2, dim=-1)
            out = out * scale + bias
        out = self.out_proj(out)


        return out