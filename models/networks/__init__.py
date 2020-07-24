from .unet_2D import *
from .unet_3D import *
from .unet_nonlocal_2D import *
from .unet_nonlocal_3D import *
from .unet_grid_attention_3D import *
from .unet_pCT_multi_att_dsv_3D import *
from .SUNETx5_3D import *


def get_network(name, n_classes, in_channels=3, feature_scale=4, tensor_dim='2D',
                nonlocal_mode='embedded_gaussian', attention_dsample=(2,2,2),
                aggregation_mode='concat'):
    model = _get_model_instance(name, tensor_dim)

    if name in ['unet']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      feature_scale=feature_scale,
                      is_deconv=False)
    elif name in ['unet_nonlocal']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      is_deconv=False,
                      nonlocal_mode=nonlocal_mode,
                      feature_scale=feature_scale)
    elif name in ['unet_grid_gating', 'unet_pct_multi_att_dsv']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      nonlocal_mode=nonlocal_mode,
                      feature_scale=feature_scale,
                      attention_dsample=attention_dsample,
                      is_deconv=False)
    elif name in ['SUNETx5']:
        # (self, in_ch=5, out_ch=2, nfilts=32, ndims=3, dropout_rate=0.2)
        model = model(in_ch=in_channels,
                      out_ch=n_classes)
    else:
        raise 'Model {} not available'.format(name)

    return model


def _get_model_instance(name, tensor_dim):
    return {
        'unet':{'2D': unet_2D, '3D': unet_3D},
        'unet_nonlocal':{'2D': unet_nonlocal_2D, '3D': unet_nonlocal_3D},
        'unet_grid_gating': {'3D': unet_grid_attention_3D},
        'unet_pct_multi_att_dsv': {'3D': unet_pCT_multi_att_dsv_3D},
        'SUNETx5': {'3D': SUNETx5}
    }[name][tensor_dim]
