import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.position_encodings import RotaryPositionEncoding2D, SinusoidalPosEmb
from utils.resnet import get_resnet, replace_bn_with_gn
from utils.layers import RelativeCrossAttentionModule
import einops
from models.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import time

class DiffusionPolicy(nn.Module):
    def __init__(self, 
                    action_dim = 2,
                    obs_horizon = 2,
                    pred_horizon = None,
                    noise_scheduler: DDPMScheduler = DDPMScheduler(),
                    vis_backbone="resnet18", 
                    device="cuda",
                    kernel_size=5,
                    cond_predict_scale=True,
                    env_size=[512, 512]):
        super().__init__()

        assert vis_backbone in ["resnet18"], "vis_backbone must be clip"


        self.obs_horizon = obs_horizon
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.device = device
        self.vision_feature_dim = 512
        self.obs_dim = self.vision_feature_dim + action_dim
        self.noise_scheduler = noise_scheduler
        self.num_inference_steps = noise_scheduler.config.num_train_timesteps

        if vis_backbone == "resnet18":
            self.vis_backbone = get_resnet("resnet18")
            self.vis_backbone = replace_bn_with_gn(self.vis_backbone)
            self.vis_backbone = self.vis_backbone.to(device)
            self.vis_backbone.eval()
            self.vis_backbone.requires_grad_(False)

        
        self.noise_pred_net = ConditionalUnet1D(input_dim=action_dim, 
                                                global_cond_dim=self.obs_dim*obs_horizon,
                                                kernel_size=kernel_size,
                                                cond_predict_scale=cond_predict_scale).to(device)
                
        self.env_size = torch.tensor(env_size).to(self.device).float()

        self.reset_time_dict()

    def reset_time_dict(self):
        self.time_dict = {
            'visual_features_t': 0,
            'noise_pred_t': 0
        }
    
    def get_time_dict(self, num_steps):
        for k in self.time_dict.keys():
            self.time_dict[k] /= num_steps
        return self.time_dict

    def compute_global_cond(self, images, agent_hist):
        """
        Args:
            images (torch.Tensor): (B, To, C, H, W)
            agent_hist (torch.Tensor): (B, To, Da)

        Returns:
            torch.Tensor: (B, *)
        """        
        # normalize data
        nimages = images

        with torch.no_grad():
            img_features = self.vis_backbone(torch.flatten(nimages, end_dim=1))
        img_features = einops.rearrange(img_features, '(b oh) c -> b oh c', oh=self.obs_horizon)
        obs_emb = torch.cat([img_features, agent_hist], dim=-1)

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

        noisy_ntraj = torch.randn(
            (B, self.pred_horizon, self.action_dim), device=self.device)
        ntraj = noisy_ntraj

        self.noise_scheduler.set_timesteps(num_inference_steps=self.num_inference_steps, device=self.device)

        for k in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                    sample=ntraj,
                    timestep=k,
                    global_cond=scene_emb.flatten(start_dim=1)
                )
            # inverse diffusion step (remove noise)
            ntraj = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=ntraj
            ).prev_sample

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
        
        # compute scene embedding
        visual_features_t = time.time()
        scene_emb = self.compute_global_cond(images, nagent_hist)
        self.time_dict['visual_features_t'] += time.time() - visual_features_t
        
        # add noise to target
        noise = torch.randn(ntraj.shape).to(self.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (B,), device=ntraj.device
        ).long()
        noisy_ntraj = self.noise_scheduler.add_noise(
            ntraj, noise, timesteps)
        
        pred_noise_t = time.time()
        noise_pred = self.noise_pred_net(sample=noisy_ntraj, timestep=timesteps, global_cond=scene_emb.flatten(start_dim=1))
        self.time_dict['noise_pred_t'] += time.time() - pred_noise_t
        loss = F.mse_loss(noise_pred, noise).mean()

        return loss
    



