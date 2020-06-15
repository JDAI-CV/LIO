import torch
import torch.nn as nn
import torch.nn.functional as F

from .coord_predict import RelativePolarCoordPredictor as CoordPredictor


class SCLModule(nn.Module):
    def __init__(self, size, feature_dim, structure_dim):
        super().__init__()
        
        self.size = size
        self.feature_dim = feature_dim
        self.structure_dim = structure_dim

        self.get_structure_feat = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.feature_dim, self.structure_dim, 1, 1),
            nn.ReLU(),
        )
        self.coord_predictor = CoordPredictor(in_dim=self.structure_dim,
                                                size=self.size)

    def forward(self, feature):
        structure_map = self.get_structure_feat(feature)
        coord_loss = self.coord_predictor(structure_map)
        return coord_loss
