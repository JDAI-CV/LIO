import torch
import torch.nn as nn
import torch.nn.functional as F

from .coord_predict import RelativeCoordPredictor as CoordPredictor


class SCLModule(nn.Module):
    def __init__(self, size, feature_dim, structure_dim, *, avg):
        super().__init__()
        
        self.size = size
        self.feature_dim = feature_dim
        self.structure_dim = structure_dim
        self.avg = avg

        self.strutureg = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.feature_dim, self.structure_dim, 1, 1),
            nn.ReLU(),
        )
        self.coord_predictor = CoordPredictor(in_dim=self.structure_dim,
                                                size=self.size)
        self.maskg = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.feature_dim, 1, 1, 1),
        )

        if self.avg:
            self.avgpool = nn.AvgPool2d(2, 2)
    
    def forward(self, feature):
        if self.avg:
            feature = self.avgpool(feature)

        mask = self.maskg(feature)
        N, _, H, W = mask.shape
        mask = mask.view(N, H, W)

        structure_map = self.strutureg(feature)
        coord_loss = self.coord_predictor(structure_map, mask)

        return feature, mask, coord_loss
