import torch
import torch.nn as nn

class DQN(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(4,128)
        self.layer2 = nn.Linear(128,128)
        self.layer3 = nn.Linear(128,2)


    def forward(self,x):

        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)

        return x