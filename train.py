import collections
import wandb
import shutil
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import copy
import matplotlib.pyplot as plt
import argparse
from skvideo.io import vwrite
from IPython.display import Video
import gdown
import os
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler

from models.ema_model import EMAModel
from policy.diffusion_transformer_image import DiffusionTransformerImage
from policy.diffusion_policy import DiffusionPolicy
from utils.environment import PushTImageEnv
from utils.dataset import PushTImageDataset, unnormalize_data
from utils.evaluate_policy import evaluate_policy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="./data/pusht_cchi_v7_replay.zarr.zip", help="path where data is stored")
    parser.add_argument('--model', type=str, default='diffusion_transformer_image', help='model to train')
    parser.add_argument('--num_epochs', type=int, default=50000, help='number of epochs to train')
    parser.add_argument('--resume_run', type=str, default=None, help='resume training from a previous run')
    parser.add_argument('--save_model_every', type=int, default=5, help='save model every n epochs')
    parser.add_argument('--n_eval', type=int, default=10, help='save model every n epochs')
    parser.add_argument('--n_episodes_eval', type=int, default=10, help='save model every n epochs')
    parser.add_argument('--embedding_dim', type=int, default=60, help='embedding dimension')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--silent', action='store_true', help='disable tqdm progress bar')
    return parser.parse_args()
    
def init_wandb(args):
    wandb_args = dict()
    wandb_args['project'] = 'T-Push'
    wandb_args['entity'] = 'felix-herrmann'
    if args.resume_run is not None:
        wandb_args['resume'] = "must"
        wandb_args['id'] = args.resume_run
    
    wandb.init(**wandb_args)
    if args.resume_run is None:
        wandb.config.update(args)

def main():
    args = parse_args()
    init_wandb(args)
    dataset_path = args.dataset_path
    model_save_dir = wandb.run.dir

    # download demonstration data from Google Drive
    if not os.path.isfile(dataset_path):
        id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
        gdown.download(id=id, output=dataset_path, quiet=False)

    # parameters
    batch_size = args.batch_size
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    #|o|o|                             observations: 2
    #| |a|a|a|a|a|a|a|a|               actions executed: 8
    #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

    # create dataset from file
    dataset = PushTImageDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon
    )
    # save training data statistics (min, max) for each dim
    stats = dataset.stats

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )


    num_epochs = args.num_epochs
    embedding_dim = args.embedding_dim
    action_dim = 2
    device = torch.device('cuda')
    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )
    obs_horizon = 2

    if args.model == 'diffusion_transformer_image':
        model = DiffusionTransformerImage(action_dim=action_dim,
                                        obs_horizon=obs_horizon,
                                        pred_horizon=pred_horizon,
                                        noise_scheduler=noise_scheduler,
                                        vis_backbone='resnet50',
                                        re_cross_attn_layer_within=5,
                                        re_cross_attn_num_heads_within=5,
                                        re_cross_attn_layer_across=5,
                                        re_cross_attn_num_heads_across=5,
                                        kernel_size=5,
                                        cond_predict_scale=True,
                                        embedding_dim=embedding_dim,
                                        device='cuda')
    elif args.model == 'diffusion_policy':
        model = DiffusionPolicy(action_dim=action_dim,
                                obs_horizon=obs_horizon,
                                pred_horizon=pred_horizon,
                                noise_scheduler=noise_scheduler,
                                vis_backbone='resnet18',
                                kernel_size=5,
                                cond_predict_scale=True,
                                device='cuda')
    model = model.to(device)


    ema_model = copy.deepcopy(model)
    _ = model.to(device)
    _ = ema_model.to(device)
    ema = EMAModel(ema_model)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=1e-4, weight_decay=1e-6)

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )
     
    start_epoch = 0
    best_loss = np.inf

    if args.resume_run is not None:
        ckptf = wandb.restore('latest_model.ckpt')
        if ckptf is not None:
            print('Resuming from run: ', args.resume_run)   
            ckpt = torch.load(ckptf.name)
            start_epoch = ckpt['epoch_idx'] + 1
            model.load_state_dict(ckpt['model'])
            best_loss = ckpt['loss']
            ema_model.load_state_dict(ckpt['ema_model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        else:
            print('No checkpoint found, starting from scratch')

    print('Start training from epoch: ', start_epoch)
    log_buffer = list()
    with tqdm(range(start_epoch, num_epochs), desc='Epoch') as tglobal:
        losses = list()
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False, disable=args.silent) as tepoch:
                for item in tepoch:
                    loss = model.compute_loss(item)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(model)

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            epoch_loss = np.mean(epoch_loss)
            losses.append(epoch_loss)
            tglobal.set_postfix(loss=epoch_loss)
            metrics = {'loss': epoch_loss}
            metrics.update(model.get_time_dict(len(dataloader)))
            log_buffer.append(metrics)
            
            if epoch_idx % args.save_model_every == 0:
                for metrics in log_buffer:
                    wandb.log(metrics)
                log_buffer = list()
                print('Saving model...')
                ckpt = {
                    'model': model.state_dict(),
                    'ema_model': ema.averaged_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch_idx': epoch_idx,
                    'loss': epoch_loss
                }
                torch.save(ckpt, os.path.join(model_save_dir, 'latest_model.ckpt'))
                wandb.save(os.path.join(model_save_dir, 'latest_model.ckpt'))
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    shutil.copyfile(os.path.join(model_save_dir, 'latest_model.ckpt'),
                                    os.path.join(model_save_dir, 'best_model.ckpt'))
                    wandb.save(os.path.join(model_save_dir, 'best_model.ckpt'))
            if epoch_idx % args.n_eval == 0:
                print('Evaluating model...')
                ema.averaged_model.eval()
                val = evaluate_policy(ema.averaged_model, stats, episodes=args.n_episodes_eval, max_steps=200, device=device)
                wandb.log({'val': val})
                ema.averaged_model.train()


    ema_model = ema.averaged_model

    max_steps = 200
    env = PushTImageEnv()
    # use a seed >200 to avoid initial states seen in the training dataset
    env.seed(100000)

    # get first observation
    obs, info = env.reset()

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
        [obs] * obs_horizon, maxlen=obs_horizon)
    # save visualization and rewards
    imgs = [env.render(mode='rgb_array')]
    rewards = list()
    done = False
    step_idx = 0

    with tqdm(total=max_steps, desc="Eval PushTImageEnv") as pbar:
        while not done:
            B = 1
            images = np.stack([x['image'] for x in obs_deque])
            images = torch.from_numpy(images).to(device).unsqueeze(0).float()
            agent_pos = np.stack([x['agent_pos'] for x in obs_deque])
            agent_pos = torch.from_numpy(agent_pos).to(device).unsqueeze(0).float()
            
            obs_dict = {
                'image': images,
                'agent_pos': agent_pos
            }

            # predict action
            with torch.no_grad():
                action = ema_model.predict_action(obs_dict)
                action = action.cpu().numpy()
            
            naction = action[0]
            action_pred = unnormalize_data(naction, stats['action'])
            
            # only take action_horizon number of actions
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end,:]
            # (action_horizon, action_dim)

            for i in range(len(action)):
                # stepping env
                obs, reward, done, _, info = env.step(action[i])
                # save observations
                obs_deque.append(obs)
                # and reward/vis
                rewards.append(reward)
                imgs.append(env.render(mode='rgb_array'))

                # update progress bar
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx > max_steps:
                    done = True
                if done:
                    break

    # print out the maximum target coverage
    print('Score: ', max(rewards))

    # visualize
    from IPython.display import Video
    vwrite('vis.mp4', imgs)
    Video('vis.mp4', embed=True, width=256, height=256)
    
    
if __name__ == '__main__':
    main()