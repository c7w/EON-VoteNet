import torch
import numpy as np

from utils.pc_util import batch_rotz

def augment(pc, augment):
    
    if augment is None:
        augment = []
    
    aug_shape = pc.shape[:2]
    pc2 = pc
    if "uniform" in augment:
        rad = torch.zeros(aug_shape).fill_(np.random.random() * np.pi * 2)
        rotmat = batch_rotz(rad)
        pc2 = (rotmat @ pc2.shape[:, :, :, None])[:, :, :, 0]

    elif "perturbation" in augment:
        rad - torch.zeros(aug_shape).fill((np.random.random() - 0.5) * np.pi / 6)  # ± 15 degree
        rotmat = batch_rotz(rad)
        pc2 = (rotmat @ pc2.shape[:, :, :, None])[:, :, :, 0]
        
    elif  "y-flip" in augment:
        pc2[:, :, 1] = -pc[:, :, 1]

    elif "x-flip" in augment:
        pc2[:, :, 0] = -pc[:, :, 0]


def weak_loss(end_points1, end_points2, loss_name):
    weak_loss = torch.tensor(0.)
    
    if "vote_features" in loss_name:
        f1 = end_points1["vote_features"]
        f2 = end_points2["vote_features"]
        weak_loss += (1 - torch.cosine_similarity(f1, f2, dim=2)).mean()
    
    if "seed_features" in loss_name:
        f1 = end_points1["seed_features"]
        f2 = end_points2["seed_features"]
        weak_loss += (1 - torch.cosine_similarity(f1, f2, dim=2)).mean()
        
    return weak_loss


def get_weak_loss(end_points, net, config):
    
    
    pc = end_points["point_clouds"]
    augment = ["perturbation"]
    pc2 = augment(pc, augment=augment)
    
    end_points1 = net(pc)
    end_points2 = net(pc2)
    
    loss_name = ["vote_features", "seed_features"]
    return weak_loss(end_points1, end_points2, loss_name)
