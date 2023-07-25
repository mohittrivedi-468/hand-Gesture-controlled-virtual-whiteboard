import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(63, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Dropout(p=0.5),

            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Dropout(p=0.5),

            nn.Linear(500, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Dropout(p=0.5),

            nn.Linear(200, 50),
            nn.ReLU(),

            nn.Linear(50, 3)
        )

    def forward(self, x):
        return self.layers(x)

def test():
    model = Model()
    noise = torch.randn((20, 63))
    out = model(noise)
    print(out.shape)
