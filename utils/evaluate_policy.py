from utils.environment import PushTImageEnv
from utils.dataset import PushTImageDataset, unnormalize_data
import collections
import torch
from tqdm import tqdm
import numpy as np

def evaluate_policy(model, stats, episodes=10, max_steps=200, obs_horizon=2, action_horizon=8, device='cuda'):
    env = PushTImageEnv()

    max_rewards = list()
    for i in range(episodes):
        # use a seed >200 to avoid initial states seen in the training dataset
        rng = np.random.randint(200, 10000)
        env.seed(rng)

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
                    action = model.predict_action(obs_dict)
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
        max_rewards.append(max(rewards))

    mean_max_rewards = np.mean(max_rewards)
    return mean_max_rewards