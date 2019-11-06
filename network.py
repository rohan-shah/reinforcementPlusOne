import torch.nn as nn
import torch.nn.functional as F

class plusOnePlayer(nn.Module):
    def __init__(self):
        super(plusOnePlayer, self).__init__()
        self._layers = []
        layerSize = 10*10
        self._layers.append(nn.Linear(5*5, layerSize))
        self._layers.extend([nn.Linear(layerSize, layerSize) for layerCounter in range(0, 7)])
        self._layers.append(nn.Linear(layerSize, 5*5))
        self._layers = nn.ModuleList(self._layers)

    def forward(self, x):
        x = x.view(-1, 5*5)
        for layer in self._layers:
            x = F.leaky_relu(layer(x))
        return(x)

class plusOneStateChange(nn.Module):
    def __init__(self):
        super(plusOneStateChange, self).__init__()
        self._layers = []
        layerSize = 10*10
        self._layers.append(nn.Linear(5*5 + 2, layerSize))
        self._layers.extend([nn.Linear(layerSize, layerSize) for layerCounter in range(0, 7)])
        self._layers.append(nn.Linear(layerSize, 5*5))
        self._layers = nn.ModuleList(self._layers)

    def forward(self, x):
        for layer in self._layers:
            x = F.relu(layer(x))
        return(x)

class plusOneValidMove(nn.Module):
    def __init__(self):
        super(plusOneValidMove, self).__init__()
        self._layers = []
        layerSize = 10*10
        self._layers.append(nn.Linear(5*5 + 2, layerSize))
        self._layers.extend([nn.Linear(layerSize, layerSize) for layerCounter in range(0, 7)])
        self._layers.append(nn.Linear(layerSize, 2))
        self._layers = nn.ModuleList(self._layers)

    def forward(self, x):
        for layer in self._layers:
            x = F.relu(layer(x))
        return(x)
