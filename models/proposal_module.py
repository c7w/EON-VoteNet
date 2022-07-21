# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from utils import pc_util
from pointnet2_repo.pointnet2_modules import PointnetSAModuleVotes

def decode_scores(net, end_points, num_class, num_heading_bin, num_size_cluster, mean_size_arr, mode_pose):
    
    mode_rot_mat = pc_util.batch_rotz(mode_pose)  # [B, Np, 3, 3]
    net_transposed = net.transpose(2,1) # (batch_size, 1024, 2+3+num_heading_bin*2+num_size_cluster*4)
    batch_size = net_transposed.shape[0]
    num_proposal = net_transposed.shape[1]

    objectness_scores = net_transposed[:,:,0:2]
    end_points['objectness_scores'] = objectness_scores
    
    base_xyz = end_points['aggregated_vote_xyz'] # (batch_size, num_proposal, 3)
    local_xyz_cano = net_transposed[:,:,2:5, None]
    local_xyz_world = torch.matmul(mode_rot_mat, local_xyz_cano).squeeze(-1)
    center = base_xyz + local_xyz_world # (batch_size, num_proposal, 3)
    end_points['center'] = center

    n_rot = num_heading_bin
    assert n_rot == num_heading_bin, 'our shifting algorithm assumes n_rot == n_heading_bin, but now n_rot=={}, n_heading_bin=={}'.format(n_rot, num_heading_bin)
    mode_pose_label, _ = pc_util.angle2class(mode_pose, n_rot)
    mode_pose_shift = mode_pose_label  # [B, Np]
    shifting = torch.arange(num_heading_bin)[None, None].to(mode_pose_shift.device)
    shifting = shifting.long() - mode_pose_shift[..., None].long()  # [B, Np, 24]
    shifting = shifting % num_heading_bin

    heading_scores = net_transposed[:,:,5:5+num_heading_bin]
    heading_scores = heading_scores.gather(2, shifting)

    heading_residuals_normalized = net_transposed[:,:,5+num_heading_bin:5+num_heading_bin*2]
    heading_residuals_normalized = heading_residuals_normalized.gather(2, shifting)
    end_points['heading_scores'] = heading_scores # Bxnum_proposalxnum_heading_bin
    end_points['heading_residuals_normalized'] = heading_residuals_normalized # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
    end_points['heading_residuals'] = heading_residuals_normalized * (np.pi/num_heading_bin) # Bxnum_proposalxnum_heading_bin

    size_scores = net_transposed[:,:,5+num_heading_bin*2:5+num_heading_bin*2+num_size_cluster]
    size_residuals_normalized = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster:5+num_heading_bin*2+num_size_cluster*4].view([batch_size, num_proposal, num_size_cluster, 3]) # Bxnum_proposalxnum_size_clusterx3
    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized
    end_points['size_residuals'] = size_residuals_normalized * torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)

    sem_cls_scores = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster*4:] # Bxnum_proposalx10
    end_points['sem_cls_scores'] = sem_cls_scores
    return end_points


class ProposalModule(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster,
                 mean_size_arr, num_proposal, sampling, seed_feat_dim=256):
        super().__init__() 

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes( 
                npoint=self.num_proposal,
                radius=0.3,
                nsample=16,
                mlp=[self.seed_feat_dim, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )
    
        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = torch.nn.Conv1d(128,128,1)
        self.conv2 = torch.nn.Conv1d(128,128,1)
        self.conv3 = torch.nn.Conv1d(128,2+3+num_heading_bin*2+num_size_cluster*4+self.num_class,1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

    def forward(self, xyz, features, end_points, FLAGS=None):
        """
        Args:
            xyz: (B,N,3), in world space
            features: (B,C,N), in "canonical" space
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        seed_pose = end_points['point_pose_pred_angle']  # [B, N], already zeroed all bg points
        seed_mask = end_points['seed_mask_pred']  # [B, N]

        """you need seed_pose to 
        (1) back-rotate xyz to "canonical space" as pointnet input
        (2) rotate predicted bbox to world space"""
        # Farthest point sampling (FPS) on votes
        seed_pose_ = seed_pose
        xyz, features, fps_inds, mode_pose = self.vote_aggregation(xyz, features, seed_pose_, seed_mask)
        sample_inds = fps_inds
        end_points['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- PROPOSAL GENERATION ---------
        net = F.relu(self.bn1(self.conv1(features))) 
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net) # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)

        end_points = decode_scores(net, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster,
                                   self.mean_size_arr, mode_pose)
        return end_points

class QuadProposalModule(nn.Module):
    def __init__(self,hidden_dim): 
        super().__init__()
        self.quad_scores_head = torch.nn.Conv1d(hidden_dim, 2, 1)
        self.center_head = torch.nn.Conv1d(hidden_dim, 3, 1)
        self.normal_vector_head = torch.nn.Conv1d(hidden_dim, 1, 1)
        self.size_head = torch.nn.Conv1d(hidden_dim, 2, 1)
        #self.direction_head = torch.nn.Conv1d(hidden_dim, 1, 1)
        self.conv1 = torch.nn.Conv1d(hidden_dim,hidden_dim,1)
        self.conv2 = torch.nn.Conv1d(hidden_dim,hidden_dim,1)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
    def forward(self,net,base_xyz,end_points,prefix):

        pc = end_points['point_clouds']
        batch_size = pc.shape[0]
        layout_pt_cnt = pc.shape[1]
        
        net = F.relu(self.bn1(self.conv1(net)))
        net = F.relu(self.bn2(self.conv2(net)))
        quad_scores = self.quad_scores_head(net).transpose(2, 1)  # (batch_size, num_proposal, 2)
        center = self.center_head(net).transpose(2, 1) + base_xyz # (batch_size, num_proposal, 3)
        
        size = self.size_head(net).transpose(2, 1) 
        #direction = self.direction_head(net).transpose(2, 1) 
        end_points[f'{prefix}quad_scores'] = quad_scores    
        end_points[f'{prefix}quad_center'] = center # (batch_size, num_proposal, 3)
        # end_points[f'{prefix}normal_vector'] = normal_vector
        end_points[f'{prefix}quad_size'] = size
        #end_points[f'{prefix}quad_direction'] = direction
        
        local_normals = torch.zeros((batch_size, 1024, 3)).cuda()
        for i in range(batch_size):
            point_cloud = pc[i][:, 0:3]
            
            # Randomly downsample point_cloud
            SAMPLE_CNT = 4000
            import random
            selected_idx = random.sample(range(layout_pt_cnt), SAMPLE_CNT)
            point_cloud = point_cloud[selected_idx, :]
            
            pc_center = point_cloud.mean(dim=0)  # (3, )
            # Normal similarity: calc and statistic analysis
            import open3d as o3d
            param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.20, max_nn=20)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud.cpu().numpy())
            pcd.estimate_normals(search_param=param)
            normals = np.asarray(pcd.normals)  # (SAMPLE_CNT, 3)
            
            reverse_mask = ((point_cloud - pc_center).cpu().numpy().reshape(SAMPLE_CNT, 1, 3) \
                    @ normals.reshape(SAMPLE_CNT, 3, 1)).reshape(SAMPLE_CNT) < 0
            
            normals[reverse_mask] = -normals[reverse_mask]
            normals = - normals
            
            from sklearn.neighbors import NearestNeighbors
            neigh = NearestNeighbors(n_neighbors=10)
            neigh.fit(point_cloud.cpu().numpy())
            neigh_dist, neigh_ind = neigh.kneighbors(end_points['quad_center'][i].detach().cpu().numpy())
            
            # neigh_ind (1024, 10)
            # normals (SAMPLE_CNT, 3)
            selected_normals = np.stack([normals[neigh_ind[i], :].mean(axis=0) for i in range(1024)], axis=0)
            selected_normals[:, 2] = 0.
            selected_normals = selected_normals / np.linalg.norm(selected_normals, axis=1)[:, None]
            selected_normals[np.isnan(selected_normals)] = 1e-6
            
            local_normals[i] = torch.tensor(selected_normals).cuda()

        assert not torch.isnan(local_normals).any()
        # delta_rad = self.normal_vector_head(net)[:, 0, :]
        # from pc_util import batch_rotz
        # normal_vector = (batch_rotz(delta_rad) @ local_normals[:,:,:,None])[..., 0]
        
        # normal_vector = .transpose(2, 1) 
        # normal_vector_norm = torch.norm(normal_vector, p=2)
        # normal_vector = normal_vector.div(normal_vector_norm)
        end_points[f'{prefix}normal_vector'] = local_normals
        
        
        return center, size, end_points

if __name__=='__main__':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, DC
    net = ProposalModule(DC.num_class, DC.num_heading_bin,
        DC.num_size_cluster, DC.mean_size_arr,
        128, 'seed_fps').cuda()
    end_points = {'seed_xyz': torch.rand(8,1024,3).cuda()}
    out = net(torch.rand(8,1024,3).cuda(), torch.rand(8,256,1024).cuda(), end_points)
    for key in out:
        print(key, out[key].shape)
