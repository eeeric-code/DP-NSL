import importlib
from .conv_based_encoder2d import build_rdn2d, build_rescnn2d, build_srresnet2d
from .conv_based_encoder3d import build_rdn3d, build_rescnn3d, build_srresnet3d
from .edsr import build_edsr2d, build_edsr3d
from .multi_scale_rescnn import build_multi_scale_rescnn



def build_encoder(args):
    if (args.encoder_name).lower() == 'rdn2d':
        return build_rdn2d(args)
    elif (args.encoder_name).lower() == 'rescnn2d':
        return build_rescnn2d(args)
    elif (args.encoder_name).lower() == 'srresnet2d':
        return build_srresnet2d(args)
    elif (args.encoder_name).lower() == 'edsr2d':
        return build_edsr2d(args)
    elif (args.encoder_name).lower() == 'rdn3d':
        return build_rdn3d(args)
    elif (args.encoder_name).lower() == 'rescnn3d':
        return build_rescnn3d(args)
    elif (args.encoder_name).lower() == 'srresnet3d':
        return build_srresnet3d(args)
    elif (args.encoder_name).lower() == 'edsr3d':
        return build_edsr3d(args)
    else:
        raise ValueError(f"Encoder {args.encoder_name} is not supported. Please choose from 'rdn2d', 'rescnn2d', 'srresnet2d', 'edsr2d', 'rdn3d', 'rescnn3d', 'srresnet3d', or 'edsr3d'.")