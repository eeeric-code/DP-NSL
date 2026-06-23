import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act=nn.ReLU(inplace=True)):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = act

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out += residual
        return out

class ResidualGroup(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=3, act=nn.ReLU(inplace=True)):
        super(ResidualGroup, self).__init__()
        self.blocks = nn.ModuleList(
            [ResidualBlock(in_channels if i == 0 else out_channels, out_channels, act) for i in range(num_blocks)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class MultiScaleResCNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_scales=3, num_blocks=6):
        super(MultiScaleResCNN, self).__init__()
        self.num_scales = num_scales
        self.head = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        encoder_layers = [ResidualGroup(out_channels*(2**i), out_channels*(2**i),num_blocks) \
                          for i in range(num_scales)]
        self.encoder = nn.ModuleList(encoder_layers)

        down_sample_layers = [nn.Conv2d(out_channels*(2**i), out_channels*(2**(i+1)), kernel_size=5, stride=2,padding=2) \
                              for i in range(num_scales-1)]
        down_sample_layers.append(nn.Identity())
        self.down_sample = nn.ModuleList(down_sample_layers)


    def forward(self, x):
        f = self.head(x)
        outputs = []
        for i, (encoder, down_sample) in enumerate(zip(self.encoder, self.down_sample)):
            f = encoder(f)
            outputs.append(f)
            f = down_sample(f)
        return outputs
    
def build_multi_scale_rescnn(args):
    """Build MultiScaleResCNN model based on the provided arguments."""
    return MultiScaleResCNN(
        in_channels=getattr(args, 'in_channels', 1),
        out_channels=getattr(args, 'n_feats', 64),
        num_scales=getattr(args, 'num_scales', 3),
        num_blocks=getattr(args, 'num_blocks', 6)
    )