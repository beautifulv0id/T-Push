import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.position_encodings import RotaryPositionEncoding2D, SinusoidalPosEmb
from utils.clip import load_clip
from utils.resnet import load_resnet50
from utils.layers import RelativeCrossAttentionModule
import einops
from models.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import time

class DiffusionTransformerImage(nn.Module):
    def __init__(self, 
                    action_dim = 2,
                    obs_horizon = 2,
                    pred_horizon = None,
                    noise_scheduler: DDPMScheduler = DDPMScheduler(),
                    vis_backbone="clip", 
                    re_cross_attn_layer_within=4, 
                    re_cross_attn_num_heads_within=4, 
                    re_cross_attn_layer_across=4,
                    re_cross_attn_num_heads_across=4,
                    embedding_dim=60, 
                    device="cuda",
                    kernel_size=5,
                    cond_predict_scale=True,
                    env_size=[512, 512]):
        super().__init__()

        assert embedding_dim % 2 == 0, "embedding_dim must be even"
        assert vis_backbone in ["clip", "resnet50"], "vis_backbone must be clip"


        self.obs_horizon = obs_horizon
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.device = device
        self.embedding_dim = embedding_dim
        self.noise_scheduler = noise_scheduler
        self.num_inference_steps = noise_scheduler.config.num_train_timesteps

        if vis_backbone == "clip":
            self.vis_backbone, self.normalize_rgb_fn = load_clip()
        elif vis_backbone == "resnet50":
            self.vis_backbone, self.normalize_rgb_fn = load_resnet50()

        self.vis_backbone = self.vis_backbone.to(device)
        self.vis_backbone.eval()
        self.vis_backbone.requires_grad_(False)
        self.vis_out_proj = nn.Linear(256, embedding_dim).to(device)


        self.query_emb = nn.Embedding(1, embedding_dim).to(device)
        self.rotary_embedder = RotaryPositionEncoding2D(embedding_dim).to(device)

        self.re_cross_attn_within = RelativeCrossAttentionModule(embedding_dim,
                                                                re_cross_attn_num_heads_within,
                                                                re_cross_attn_layer_within).to(device)
        self.re_cross_attn_across = RelativeCrossAttentionModule(embedding_dim,
                                                                re_cross_attn_num_heads_across,
                                                                re_cross_attn_layer_across).to(device)
        
        self.noise_pred_net = ConditionalUnet1D(input_dim=action_dim, 
                                                global_cond_dim=embedding_dim,
                                                kernel_size=kernel_size,
                                                cond_predict_scale=cond_predict_scale).to(device)
                
        self.time_embedder = SinusoidalPosEmb(embedding_dim).to(device)

        self.env_size = torch.tensor(env_size).to(self.device).float()

        self.reset_time_dict()

    def reset_time_dict(self):
        self.time_dict = {
            'visual_features_t': 0,
            'local_obs_attention_t': 0,
            'global_obs_attention_t': 0,
            'noise_pred_t': 0
        }
    
    def get_time_dict(self, num_steps):
        for k in self.time_dict.keys():
            self.time_dict[k] /= num_steps
        return self.time_dict

    def compute_visual_features(self, rgb, out_res=[24, 24]):
        out = self.vis_backbone(rgb)["res2"]
        h, w = out.shape[-2:]
        out = einops.rearrange(out, 'b c h w -> b (h w) c')
        out = self.vis_out_proj(out)
        out = einops.rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        return out
    
    def get_img_positions(self, shape):
        yx = torch.meshgrid([torch.linspace(-1, 1, shape[0]), torch.linspace(-1, 1, shape[1])], indexing='ij')
        yx = torch.stack(yx, dim=0).float().to(self.device)
        return yx
    
    def rotary_embed(self, x):
        """
        Args:
            x (torch.Tensor): (B, N, Da)
        Returns:
            torch.Tensor: (B, N, 2, D//2)
        """
        return self.rotary_embedder(x)
    
    def compute_context_features(self, rgb):
        """
        Args:
            rgb (torch.Tensor): (B, C, H, W)
        Returns:
            torch.Tensor: (B, N, C)
            torch.Tensor: (B, N, 2, D//2)
        """
        visual_features_t = time.time()
        rgb_features = self.compute_visual_features(rgb)
        self.time_dict['visual_features_t'] += time.time() - visual_features_t
        rgb_pos = self.get_img_positions(rgb_features.shape[-2:])
        rgb_pos = einops.repeat(rgb_pos, 'xy h w -> b (h w) xy', b=rgb.shape[0])
        context_features = einops.repeat(rgb_features, 'b c h w -> (h w) b c')
        context_pos = self.rotary_embed(rgb_pos)
        return context_features, context_pos
    
    def compute_query_features(self, agent_pos):
        query = einops.repeat(self.query_emb.weight, '1 c -> 1 b c', b=agent_pos.shape[0])
        query_pos = einops.repeat(agent_pos, 'b c -> b 1 c')
        query_pos = self.rotary_embed(query_pos)
        return query, query_pos
    
    def compute_local_obs_representation(self, context_features, context_pos, query, query_pos):
        obs_embs = self.re_cross_attn_within(query=query, value=context_features, query_pos=query_pos, value_pos=context_pos)
        obs_embs = obs_embs[-1].squeeze(0)
        return obs_embs
    
    def attend_across_obs(self, obs_embs):
        """
        Args:
            obs_embs (torch.Tensor): (oh, B, C)
        Returns:
            torch.Tensor: (B, C)
        """
        # add time embedding
        time_emb = self.time_embedder(torch.arange(self.obs_horizon).to(self.device)).unsqueeze(1)
        obs_embs = obs_embs + time_emb

        # cross attent (temporally) across observations
        obs_emb = self.re_cross_attn_across(value=obs_embs[:-1], query=obs_embs[-1:])
        obs_emb = obs_emb[-1].squeeze(0)
        return obs_emb
        
    def to_rel_trajectory(self, traj, agent_pos):
        """
        Args:
            traj (torch.Tensor): (B, t, Da)
            agent_pos (torch.Tensor): (B, Da)

        Returns:
            torch.Tensor: (B, t, Da)
        """
        traj = traj - agent_pos[:, None, :]
        return traj

    def to_abs_trajectory(self, traj, agent_pos):
        """
        Args:
            traj (torch.Tensor): (B, t, Da)
            agent_pos (torch.Tensor): (B, Da)

        Returns:    
            torch.Tensor: (B, t, Da)
        """
        traj = traj + agent_pos[:, None, :]
        return traj

    def compute_global_cond(self, images, agent_hist):
        """
        Args:
            images (torch.Tensor): (B, To, C, H, W)
            agent_hist (torch.Tensor): (B, To, Da)

        Returns:
            torch.Tensor: (B, *)
        """
        # rearrange data s.t. each observation is encoded separately
        images = einops.rearrange(images, 'b oh c h w -> (b oh) c h w')
        agent_hist = einops.rearrange(agent_hist, 'b oh c -> (b oh) c')
        
        # normalize data
        nimages = self.normalize_rgb_fn(images)

        # compute scene embedding
        context_features, context_pos = self.compute_context_features(nimages)
        query, query_pos = self.compute_query_features(agent_hist)

        local_obs_attention_t = time.time()
        obs_embs = self.compute_local_obs_representation(context_features, context_pos, query, query_pos)
        self.time_dict['local_obs_attention_t'] += time.time() - local_obs_attention_t

        obs_embs = einops.rearrange(obs_embs, '(b oh) c -> oh b c', oh=self.obs_horizon)

        global_obs_attention_t = time.time()
        obs_emb = self.attend_across_obs(obs_embs)
        self.time_dict['global_obs_attention_t'] += time.time() - global_obs_attention_t

        return obs_emb

    def predict_action(self, obs_dict) -> torch.Tensor:
        """
        Args:
            obs_dict (dict): dict of observations
                image (torch.Tensor): (B, To, C, H, W)
                agent_pos (torch.Tensor): (B, To, Da)
        Returns:    
            torch.Tensor: (B, Ta, Da)
        """

        # get observation data
        images = obs_dict["image"].to(self.device)
        agent_hist = obs_dict["agent_pos"].to(self.device)
        B = images.shape[0]

        # already normalized to [-1, 1]
        nagent_hist = agent_hist

        # compute scene embedding
        scene_emb = self.compute_global_cond(images, nagent_hist)

        noisy_nrtraj = torch.randn(
            (B, self.pred_horizon, self.action_dim), device=self.device)
        nrtraj = noisy_nrtraj

        self.noise_scheduler.set_timesteps(num_inference_steps=self.num_inference_steps, device=self.device)

        for k in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                    sample=nrtraj,
                    timestep=k,
                    global_cond=scene_emb
                )
            # inverse diffusion step (remove noise)
            nrtraj = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=nrtraj
            ).prev_sample

        ntraj = self.to_abs_trajectory(nrtraj, agent_hist[:, -1])

        return ntraj

    def compute_loss(self, batch):
        """
        Args:
            batch (dict): dict of observations
                image (torch.Tensor): (B, To, C, H, W)
                agent_pos (torch.Tensor): (B, To, Da)
                action (torch.Tensor): (B, Ta, Da)
        Returns:    
            torch.Tensor: (B, Ta, Da)
        """
        # get batch data
        images = batch["image"].to(self.device)
        agent_hist = batch["agent_pos"].to(self.device)
        traj = batch["action"].to(self.device)

        B = agent_hist.shape[0]

        # already normalized to [-1, 1]
        ntraj = traj
        nagent_hist = agent_hist
        
        # convert to relative trajectory
        nrtraj = self.to_rel_trajectory(ntraj, nagent_hist[:, -1])


        # compute scene embedding
        scene_emb = self.compute_global_cond(images, nagent_hist)

        # add noise to target
        noise = torch.randn(nrtraj.shape).to(self.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (B,), device=nrtraj.device
        ).long()
        noisy_nrtraj = self.noise_scheduler.add_noise(
            nrtraj, noise, timesteps)
        
        noise_pred_t = time.time()
        noise_pred = self.noise_pred_net(sample=noisy_nrtraj, timestep=timesteps, global_cond=scene_emb)
        self.time_dict['noise_pred_t'] += time.time() - noise_pred_t

        loss = F.mse_loss(noise_pred, noise).mean()

        return loss
    



