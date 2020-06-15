import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class RelativePolarCoordPredictor(nn.Module):
    def __init__(self, in_dim, size=7):
        super().__init__()
        self.in_dim = in_dim
        self.size = size

        self.predictor = nn.Sequential(
                nn.Linear(2*in_dim, 2),
                nn.ReLU(),
        )
        
        self.basic_label = torch.from_numpy(self.build_basic_label(size)).float()
        self.dist_loss_f = nn.MSELoss(reduction='none')

    def forward(self, x):
        N, C, H, W = x.shape
        reduced_x = x.view(N, C, H*W).transpose(1, 2).contiguous()  # (N, S, C)

        _, reduced_x_max_index = torch.max(torch.mean(reduced_x, dim=-1), dim=-1)

        basic_index = torch.from_numpy(np.array([i for i in range(N)])).cuda()
        max_features = reduced_x[basic_index, reduced_x_max_index, :]  # (N, C)
        max_features_to_concat = max_features.unsqueeze(1).expand((N, H*W, C))

        discriminative_feature = torch.cat((max_features_to_concat, reduced_x), dim=-1)  # (N, S, 2*C)
        discriminative_feature = discriminative_feature.view(N*H*W, 2*C)
        preds_coord = self.predictor(discriminative_feature)  # (N*H*W, 2)
        preds_coord = preds_coord.view(N, H*W, 2)
        preds_coord[basic_index, reduced_x_max_index, :] = 0.
        preds_coord = preds_coord.view(N, H, W, 2)

        # Build Label
        label = self.basic_label.cuda()
        label = label.unsqueeze(0).expand((N, H, W, 2)).view(N, H*W, 2)  # (N, S, 2)
        basic_anchor = label[basic_index, reduced_x_max_index, :].unsqueeze(1)  # (N, 2)
        relative_coord = label - basic_anchor
        relative_coord = relative_coord / self.size
        relative_dist = torch.sqrt(torch.sum(relative_coord**2, dim=-1))  # (N, S)
        relative_angle = torch.atan2(relative_coord[:, :, 1], relative_coord[:, :, 0])  # (N, S) in (-pi, pi)
        relative_angle = (relative_angle / np.pi + 1) / 2  # (N, S) in (0, 1)

        # Calc Loss
        preds_dist, preds_angle = preds_coord[:, :, :, 0], preds_coord[:, :, :, 1]

        preds_dist = preds_dist.view(N, H, W)
        relative_dist = relative_dist.view(N, H, W)
        dist_loss = self.dist_loss_f(preds_dist, relative_dist)

        preds_angle = preds_angle.view(N, H*W)
        gap_angle = preds_angle - relative_angle  # (N, S) in (0, 1) - (0, 1) = (-1, 1)
        gap_angle[gap_angle < 0] += 1
        gap_angle = gap_angle - torch.mean(gap_angle, dim=-1, keepdim=True)  # (N, H*W)
        gap_angle = gap_angle.view(N, H, W)
        angle_loss = torch.pow(gap_angle, 2)
        return dist_loss + angle_loss
    
    def build_basic_label(self, size):
        if isinstance(size, int):
            basic_label = np.array([[(i, j) for j in range(size)] for i in range(size)])
        elif (isinstance(size, list) or isinstance(size, tuple)) and len(size) == 2:
            h = size[0]
            w = size[1]
            basic_label = np.array([[(i, j) for j in range(w)] for i in range(h)])
            if h == w:
                self.size = h
        return basic_label

