import numpy as np
import cv2

def get_masks(imgs):
    imgs = imgs.reshape(-1, *imgs.shape[-3:])
    masks = {
        'object': [],
        'goal': [],
        'robot': []
    }
  
    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        object_mask = (img[:,:,1] < 0.5) & (img[:,:,2] < 0.8)
        goal_mask = (img[:,:,0] >= 66) & (img[:,:,0] < 185) & (img[:,:,1] > 0.1) & (img[:,:,2] > 0.8)
        robot_mask = (img[:,:,0] >= 185) & (img[:,:,0] < 256) & (img[:,:,1] > 0.1) & (img[:,:,2] > 0.8)
        masks['object'].append(object_mask)
        masks['goal'].append(goal_mask)
        masks['robot'].append(robot_mask)

    masks['object'] = np.stack(masks['object'], axis=0)
    masks['goal'] = np.stack(masks['goal'], axis=0)
    masks['robot'] = np.stack(masks['robot'], axis=0)

    return masks