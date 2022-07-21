import random
import torch
import numpy as np

from utils.pc_util import batch_rotz

def do_augment(pc, augment):
    
    if augment is None:
        augment = []

    aug_shape = pc.shape[:2]
    pc2 = pc[:, :, 0:3]
    if "uniform" in augment:
        rad = torch.zeros(aug_shape).fill_(np.random.random() * np.pi * 2)
        rotmat = batch_rotz(rad).to(pc.device)
        pc2 = (rotmat @ pc2[:, :, :, None])[:, :, :, 0]

    if "orthogonal" in augment:
        ind, rad = random.choice([(0, 0), (1, np.pi / 2), (2, np.pi), (3, np.pi / 2 * 3)])
        rad = torch.zeros(aug_shape).fill_(rad)
        rotmat = batch_rotz(rad).to(pc.device)
        pc2 = (rotmat @ pc2[:, :, :, None])[:, :, :, 0]

    if "perturbation" in augment:
        rad - torch.zeros(aug_shape).fill_((np.random.random() - 0.5) * np.pi / 6)  # ± 15 degree
        rotmat = batch_rotz(rad).to(pc.device)
        pc2 = (rotmat @ pc2[:, :, :, None])[:, :, :, 0]
        
    if "y-flip" in augment:
        pc2[:, :, 1] = -pc[:, :, 1]

    if "x-flip" in augment:
        pc2[:, :, 0] = -pc[:, :, 0]
    
    pc2 = torch.concat([pc2, pc[:, :, 3:]], dim=2)
    return pc2


def weak_loss(end_points1, end_points2, loss_name):
    weak_loss = torch.tensor(0.).cuda()
    
    if "vote_features" in loss_name:
        f1 = end_points1["vote_features"]
        f2 = end_points2["vote_features"]
        weak_loss += (1 - torch.cosine_similarity(f1, f2, dim=2)).abs().mean()
    
    if "seed_features" in loss_name:
        f1 = end_points1["seed_features"]
        f2 = end_points2["seed_features"]
        weak_loss += (1 - torch.cosine_similarity(f1, f2, dim=2)).abs().mean()
        
    return weak_loss


def get_weak_loss(end_points, net, config, augment=["perturbation", "orthogonal"], loss_name=["vote_features", "seed_features"]):
    
    
    pc = end_points["point_clouds"]
    
    pc2 = do_augment(pc, augment=augment)
    
    end_points1 = net({"point_clouds": pc})
    
    with torch.no_grad():
        end_points2 = net({"point_clouds": pc2})
    
    loss_name = ["vote_features", "seed_features"]
    return weak_loss(end_points1, end_points2, loss_name)

def get_weak_loss_mean_teacher(end_points, net, ema_net, augment=["perturbation", "orthogonal"], loss_name=["vote_features", "seed_features"]):
    f1 = net(end_points)
    with torch.no_grad():
        f2 = ema_net(end_points)
        
    weak_loss = torch.tensor(0.).cuda()
    for key in f1.keys():
        if 'feature' in key:
            weak_loss += ((f1[key] - f2[key]) ** 2).mean()
    
    return weak_loss