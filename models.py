import torch.nn as nn

class MLP(nn.Module):
    """Compact MLP for vector inputs."""
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )
    def forward(self, x):  # x: [B, d_in]
        return self.net(x)

def build_mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Module:
    return MLP(d_in=d_in, d_hidden=d_hidden, d_out=d_out)