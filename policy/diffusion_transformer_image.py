import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.position_encodings import RotaryPositionEncoding2D, SinusoidalPosEmb
from utils.clip import load_clip
from utils.layers import RelativeCrossAttentionModule
import einops
from models.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class DiffusionTransformerImage(nn.Module):
    def __init__(self, 
                    action_dim = 2,
                    obs_horizon = 2,
                    action_horizon = None,
                    noise_scheduler: DDPMScheduler = DDPMScheduler(),
                    vis_backbone="clip", 
                    re_cross_attn_layer=4, 
                    re_cross_attn_num_heads=4, 
                    embedding_dim=60, 
                    device="cuda",
                    env_size=[512, 512]):
        super().__init__()

        assert embedding_dim % 2 == 0, "embedding_dim must be even"
        assert vis_backbone in ["clip", "resnet50"], "vis_backbone must be either clip or resnet50"


        self.obs_horizon = obs_horizon
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.device = device
        self.embedding_dim = embedding_dim
        self.noise_scheduler = noise_scheduler
        self.num_inference_steps = noise_scheduler.config.num_train_timesteps

        if vis_backbone == "clip":
            self.vis_backbone, self.normalize_rgb_fn = load_clip()
            self.vis_backbone = self.vis_backbone.to(device)
            self.vis_backbone.eval()
            self.vis_backbone.requires_grad_(False)
            self.vis_out_proj = nn.Linear(256, embedding_dim).to(device)

        self.query_embedding = nn.Embedding(1, embedding_dim).to(device)
        self.rotary_embedder = RotaryPositionEncoding2D(embedding_dim).to(device)

        self.re_cross_attn = RelativeCrossAttentionModule(embedding_dim,
                                                                re_cross_attn_num_heads,
                                                                re_cross_attn_layer).to(device)
        
        self.noise_pred_net = ConditionalUnet1D(input_dim=action_dim, global_cond_dim=embedding_dim)
        
        self.time_embedder = SinusoidalPosEmb(embedding_dim).to(device)

        self.env_size = torch.tensor(env_size).to(self.device).float()
        self.normalize_pos_fn = lambda pos: pos / self.env_size
        self.unnormalize_pos_fn = lambda pos: pos * self.env_size

    def compute_visual_features(self, rgb, out_res=[24, 24]):
        with torch.no_grad():
            if out_res == [24, 24]:
                out = self.vis_backbone(rgb)["res2"]
            elif out_res == [48, 48]:
                out = self.vis_backbone(rgb)["res1"]
            else:
                raise NotImplementedError
        h, w = out.shape[-2:]
        out = einops.rearrange(out, 'b c h w -> b (h w) c')
        out = self.vis_out_proj(out)
        out = einops.rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        out = F.interpolate(out, size=out_res, mode='bilinear', align_corners=False)
        return out
    
    def get_img_positions(self, shape):
        xy = torch.meshgrid([torch.linspace(0, 1, shape[0]), torch.linspace(0, 1, shape[1])])
        xy = torch.stack(xy, dim=0).float().to(self.device)
        return xy
    
    def rotary_embed(self, x):
        return self.rotary_embedder(x)
    
    def compute_context_features(self, rgb):
        rgb_features = self.compute_visual_features(rgb)
        rgb_pos = self.get_img_positions(rgb_features.shape[-2:])
        rgb_pos = einops.repeat(rgb_pos, 'c h w -> b (h w) c', b=rgb.shape[0])
        context_features = einops.repeat(rgb_features, 'b c h w -> (h w) b c')
        context_pos = self.rotary_embed(rgb_pos)
        return context_features, context_pos
    
    def compute_query_features(self, agent_pos):
        query = einops.repeat(self.query_embedding.weight, '1 c -> 1 b c', b=agent_pos.shape[0])
        query_pos = einops.repeat(agent_pos, 'b c -> b 1 c')
        query_pos = self.rotary_embed(query_pos)
        return query, query_pos
    
    def compute_scene_embedding(self, context_features, context_pos, query, query_pos):
        scene_embeddings = self.re_cross_attn(query=query, value=context_features, query_pos=query_pos, value_pos=context_pos)
        scene_embeddings = scene_embeddings[-1].squeeze(0)
        scene_embeddings = einops.rearrange(scene_embeddings, '(b oh) c -> oh b c', oh=self.obs_horizon)
        scene_embedding = self.re_cross_attn(value=scene_embeddings[:-1], query=scene_embeddings[-1:])
        scene_embedding = scene_embedding[-1].squeeze(0)
        return scene_embedding

    def normalize_rgb(self, rgb):
        rgb = einops.rearrange(rgb, 'b t c h w -> (b t) c h w')
        rgb = self.normalize_rgb_fn(rgb)
        rgb = einops.rearrange(rgb, '(b t) c h w -> b t c h w', t=self.obs_horizon)
        return rgb
    
    def normalize_pos(self, pos):
        pos = self.normalize_pos_fn(pos)
        return pos
    
    def unnormalize_pos(self, pos):
        pos = self.unnormalize_pos_fn(pos)
        return pos
    
    def normalize_trajectory(self, traj, agent_pos):
        traj = traj - agent_pos[:, None, :]
        return traj

    def unnormalize_trajectory(self, traj, agent_pos):
        traj = traj + agent_pos[:, None, :]
        return traj

    def predict_action(self, obs_dict) -> torch.Tensor:
        # get observation data
        rgb = obs_dict["image"].to(self.device)
        agent_hist = obs_dict["agent_pos"].to(self.device)
        agent_pos = agent_hist[:, -1]
        B = rgb.shape[0]
        
        # normalize data
        nrgb = self.normalize_rgb_fn(rgb)
        agent_hist = self.normalize_pos(agent_hist)

        # compute scene embedding
        nrgb = einops.rearrange(nrgb, 'b oh c h w -> (b oh) c h w')
        agent_hist = einops.rearrange(agent_hist, 'b oh c -> (b oh) c')
        context_features, context_pos = self.compute_context_features(nrgb)
        query, query_pos = self.compute_query_features(agent_hist)
        scene_embedding = self.compute_scene_embedding(context_features, context_pos, query, query_pos)

        noisy_action = torch.randn(
            (B, self.action_horizon, self.action_dim), device=self.device)
        naction = noisy_action

        self.noise_scheduler.set_timesteps(num_inference_steps=self.num_inference_steps, device=self.device)

        for k in range(self.num_inference_steps):
            noise_pred = self.noise_pred_net(
                    sample=naction,
                    timestep=k,
                    global_cond=scene_embedding
                )
                        # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        action = self.unnormalize_pos(naction)
        return action


    def compute_loss(self, batch):
        # get batch data
        rgb = batch["image"].to(self.device)
        agent_hist = batch["agent_pos"].to(self.device)
        agent_pos = agent_hist[:, -1]
        traj = batch["action"].to(self.device)
        B = rgb.shape[0]

        # normalize data
        nrgb = self.normalize_rgb_fn(rgb)
        agent_hist = self.normalize_pos(agent_hist)
        traj = self.normalize_pos(traj)
        traj = self.normalize_trajectory(traj, agent_pos)

        # compute scene embedding
        nrgb = einops.rearrange(nrgb, 'b oh c h w -> (b oh) c h w')
        agent_hist = einops.rearrange(agent_hist, 'b oh c -> (b oh) c')
        context_features, context_pos = self.compute_context_features(nrgb)
        query, query_pos = self.compute_query_features(agent_hist)
        scene_embedding = self.compute_scene_embedding(context_features, context_pos, query, query_pos)

        # add noise to target
        noise = torch.randn(traj.shape).to(self.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (B,), device=traj.device
        ).long()
        noisy_traj = self.noise_scheduler.add_noise(
            traj, noise, timesteps)
        
        noise_pred = self.noise_pred_net(sample=noisy_traj, timestep=timesteps, global_cond=scene_embedding)
        loss = F.mse_loss(noise_pred, noise).mean()

        return loss
    



