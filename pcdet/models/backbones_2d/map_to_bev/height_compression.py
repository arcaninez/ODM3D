import torch.nn as nn


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                frame_id
                gt_boxes
                points
                use_lead_xyz
                voxels (num_voxels, max_points_per_voxel, 4)
                voxel_coords: (num_voxels, 4)
                voxel_num_points: (num_voxels, )
                image_shape: (batch_size, 2)
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense() # (batch_size,feature_dim=64, D=2, H=188, W=140)
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict
