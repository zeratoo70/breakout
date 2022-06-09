from copy import deepcopy
from typing import Tuple
from torch import nn

class DQN(nn.Module):
    def __init__(self, state_shape: Tuple[int, int, int],  n_actions: int) -> None:
        
        super().__init__()
        c, h, w = state_shape
        self.main = nn.Sequential(
            # input shape: c x 84 x 84
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            # input shape: 32 x 20 x 20
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            # input shape: 64 x 9 x 9
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            # input shape: 64 x 7 x 7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
        self.target = deepcopy(self.main)

        for p in self.target.parameters():
            p.requires_grad = False
    
    def forward(self, input, model):
        if model == "main":
            return self.main(input)
        elif model == "target":
            return self.target(input)
        else:
            raise ValueError("Parameter model must be either main or target")