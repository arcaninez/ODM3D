import torch
import torch.nn as nn
from pcdet.models.model_utils.basic_block_2d import BasicBlock2D

class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size # nx=432, ny=496, nz=1
        assert self.nz == 1 # nz==1 for pillars

        self.use_conv_layers = model_cfg.get('USE_CONV_LAYERS', False)
        if self.use_conv_layers:
            self.conv_layers = nn.ModuleList()
            self.conv_layer_num = model_cfg.get('CONV_LAYER_NUM', 5)
            self.use_ds_layer = model_cfg.get('USE_DS_LAYER', False)
            if self.use_ds_layer:
                ds_layer = BasicBlock2D(
                        in_channels=self.num_bev_features,
                        out_channels=self.num_bev_features,
                        kernel_size=3,padding=1,stride=2)
                self.conv_layers.append(ds_layer)
            for i in range(self.conv_layer_num):
                single_conv_layer = BasicBlock2D(
                        in_channels=self.num_bev_features,
                        out_channels=self.num_bev_features,
                        kernel_size=3,padding=1)
                self.conv_layers.append(single_conv_layer)

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords'] # pillar_features (num_pillars, num_pillar_feats); coords: (num_pillars, 4)
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros( # (num_bev_feat=64, num_pillars=1*432*496)
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx # IMPORTANT! (num_pillars,); get idx of coords of current batch
            this_coords = coords[batch_mask, :] # get coords of current batch
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3] # IMPORTANT: compute the indices
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :] # get features of current batch
            pillars = pillars.t() # (64, num_pillars)
            spatial_feature[:, indices] = pillars # spatial_feature: (64, num_pillars); use indices to fill in feature of pillars of current batch; other places are empty!
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0) # stack a list of 4 (=batch_size) to a tensor
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx) # (batch_size, 64*1, ny=492, nx=432)

        if self.use_conv_layers:
            for l in self.conv_layers:
                batch_spatial_features = l(batch_spatial_features)

        batch_dict['spatial_features'] = batch_spatial_features  # (batch_size, 64*1, ny=492, nx=432)
        return batch_dict
