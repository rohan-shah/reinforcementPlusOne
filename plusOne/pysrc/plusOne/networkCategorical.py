import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def standardiseCategorical(game, boardSize):
    board = game.getBoard()
    data = np.zeros(shape = (boardSize, boardSize, 9))
    values = (board - game.getMax() + 9)
    for x in range(0, boardSize):
        for y in range(0, boardSize):
            data[y, x, values[y, x]] = 1
    return(data)

def standardiseCategoricalWithMove(game, x, y):
    board = game.getBoard()
    data = np.zeros(shape = (boardSize, boardSize, 9))
    values = (board - game.getMax() + 9)
    for x in range(0, boardSize):
        for y in range(0, boardSize):
            data[y, x, values[y, x]] = 1
    data[y, x, :] = -1 * data[y, x, :]
    return(data)

def markMoveInCategorical(standardised, x, y):
    standardised = standardised.copy()
    standardised[y, x, :] = -1 * standardised[y, x, :]
    return standardised

def standardiseCategoricalSimulated(board, max):
    data = np.zeros(shape = (boardSize, boardSize, 9))
    values = (board - max + 9)
    for x in range(0, boardSize):
        for y in range(0, boardSize):
            data[y, x, values[y, x]] = 1
    return(data)

def categoricalToBoard(encoded, max):
    result = np.apply_along_axis(np.argmax, 2, encoded)
    result = result - np.max(result) + max - 2
    return(result)

class plusOneValidMove(nn.Module):
    def __init__(self, nLayers, boardSize, nExtra):
        super(plusOneValidMove, self).__init__()
        self._nLayers = nLayers
        self._layers = []
        layerSize = (boardSize + nExtra) ** 2
        self._layers.append(nn.Linear(boardSize*boardSize*9, layerSize))
        self._layers.extend([nn.Linear(layerSize, layerSize) for layerCounter in range(0, nLayers)])
        self._layers.append(nn.Linear(layerSize, 2))
        self._layers = nn.ModuleList(self._layers)

    def forward(self, x):
        for layer in self._layers:
            x = F.leaky_relu(layer(x))
        return(x)

class plusOneValidMoveAll(nn.Module):
    def __init__(self, nLayers, boardSize, nExtra):
        super(plusOneValidMoveAll, self).__init__()
        self._nLayers = nLayers
        self._layers = []
        layerSize = (boardSize + nExtra) ** 2
        self._layers.append(nn.Linear(boardSize*boardSize*9, layerSize))
        self._layers.extend([nn.Linear(layerSize, layerSize) for layerCounter in range(0, nLayers)])
        self._layers.append(nn.Linear(layerSize, boardSize*boardSize*2))
        self._layers = nn.ModuleList(self._layers)
        self._boardSize = boardSize

    def forward(self, x):
        for layer in self._layers:
            x = F.leaky_relu(layer(x))
        return(x.view(self._boardSize*self._boardSize, 2))

