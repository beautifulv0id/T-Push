import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.position_encodings import RotaryPositionEncoding2D, SinusoidalPosEmb
from utils.clip import load_clip
from utils.layers import RelativeCrossAttentionModule
import einops
from models.film import FILM
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class DiffusionTransformerImage(nn.Module):
    def __init__(self, 
                    action_dim = 2,
                    obs_horizon = 2,
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
        
        self.noise_pred_net = FILM(
                                in_channels=action_dim,  
                                cond_channels=embedding_dim,
                                hidden_channels=512,
                                diffusion_step_embed_dim=256
                                ).to(device)
        
        self.time_embedder = SinusoidalPosEmb(embedding_dim).to(device)

        self.glob_scene_encoder = nn.Sequential(
            nn.Linear(2*embedding_dim, 2*embedding_dim),
            nn.ReLU(),
            nn.Linear(2*embedding_dim, 2*embedding_dim),
            nn.ReLU(),
            nn.Linear(2*embedding_dim, embedding_dim),
            nn.ReLU(),
        )

        self.env_size = torch.tensor(env_size).to(self.device).float()
        self.normalize_pos_fn = lambda pos: pos / self.env_size
        self.unnormalize_pos_fn = lambda pos: pos * self.env_size

    def compute_visual_features(self, rgb, out_res=[24, 24]):
        rgb = einops.rearrange(rgb, 'b t c h w -> (b t) c h w')
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
        out = einops.rearrange(out, '(b t) c h w -> b t c h w', t=self.obs_horizon)
        return out
    
    def get_img_positions(self, shape):
        xy = torch.meshgrid([torch.linspace(0, 1, shape[0]), torch.linspace(0, 1, shape[1])])
        xy = torch.stack(xy, dim=0).float().to(self.device)
        return xy
    
    def rotary_embed(self, x):
        return self.rotary_embedder(x)
    
    def compute_context_features(self, rgb):
        rgb_features = self.compute_visual_features(rgb)
        # (b, t, c, h, w)
        time_embed = self.time_embedder(torch.arange(self.obs_horizon).to(self.device))
        time_embed = einops.repeat(time_embed, 't c -> b t c h w', b=rgb_features.shape[0], h=rgb_features.shape[-2], w=rgb_features.shape[-1])
        context_features = rgb_features + time_embed
        context_features = einops.repeat(rgb_features, 'b t c h w -> (t h w) b c')
        
        rgb_pos = self.get_img_positions(rgb_features.shape[-2:])
        rgb_pos = einops.repeat(rgb_pos, 'c h w -> b (t h w) c', b=rgb_features.shape[0], t=self.obs_horizon)
        context_pos = self.rotary_embed(rgb_pos)
        return context_features, context_pos
    
    def compute_query_features(self, agent_pos):
        # should I use the same query for all timesteps?
        query = einops.repeat(self.query_embedding.weight, '1 c -> t b c', b=agent_pos.shape[0], t=self.obs_horizon)
        query_pos = self.rotary_embed(agent_pos)
        return query, query_pos
    
    def compute_scene_embedding(self, context_features, context_pos, query, query_pos):
        scene_embedding = self.re_cross_attn(query=query, value=context_features, query_pos=query_pos, value_pos=context_pos)
        scene_embedding = scene_embedding[-1].squeeze(0)
        scene_embedding = einops.rearrange(scene_embedding, 't b c -> b (t c)')
        return scene_embedding

    def predict_action(self, obs_dict) -> torch.Tensor:
        # get observation data
        abs_agent_pos = obs_dict["agent_pos"].to(self.device)
        rgb = obs_dict["rgb"].to(self.device)
        B = rgb.shape[0]
        
        # normalize data
        nrgb = self.normalize_rgb(rgb)
        nagent_pos = self.normalize_pos(abs_agent_pos)

        context_features, context_pos = self.compute_context_features(nrgb)
        query, query_pos = self.compute_query_features(nagent_pos)
        scene_embeddings = self.compute_scene_embedding(context_features, context_pos, query, query_pos)
        # should maybe be replaced by transformer or find different way to aggregate
        scene_embedding = self.glob_scene_encoder(scene_embeddings)

        noisy_action = torch.randn(
            (B, self.action_dim), device=self.device)
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

        action = naction + nagent_pos
        action = self.unnormalize_pos(action)
        return action
    
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

    def compute_loss(self, batch):
        # get batch data
        rgb = batch["rgb"].to(self.device)
        abs_agent_pos = batch["agent_pos"].to(self.device)
        abs_goal_pos = batch["goal_pose"][...,:2].to(self.device)
        B = rgb.shape[0]

        # normalize data
        nrgb = self.normalize_rgb_fn(rgb)
        nagent_pos = self.normalize_pos(abs_agent_pos)
        ngoal_pos = self.normalize_pos(abs_goal_pos)

        context_features, context_pos = self.compute_context_features(nrgb)
        query, query_pos = self.compute_query_features(nagent_pos)
        scene_embeddings = self.compute_scene_embedding(context_features, context_pos, query, query_pos)
        # should maybe be replaced by transformer or find different way to aggregate
        scene_embedding = self.glob_scene_encoder(scene_embeddings)

        # add noise to target
        goal_pos = ngoal_pos[:,-1] - nagent_pos[:,-1]
        noise = torch.randn(goal_pos.shape).to(self.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (goal_pos.shape[0],), device=goal_pos.device
        ).long()
        noisy_goal_pos = self.noise_scheduler.add_noise(
            goal_pos, noise, timesteps)
        
        pred = self.noise_pred_net(sample=noisy_goal_pos, timestep=timesteps, global_cond=scene_embedding)
        loss = F.mse_loss(pred, goal_pos).mean()

        return loss
    



