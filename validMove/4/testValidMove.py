import torch
import plusOne
from plusOne.networkCategorical import plusOneValidMove, standardiseCategorical, standardiseCategoricalSimulated, categoricalToBoard, markMoveInCategorical
boardSize = 4
currentGame = plusOne.libplusOne_python.Game(boardSize)
import numpy as np
import time

startingState = currentGame.getBoard()

model = plusOneValidMove(nLayers = 7, boardSize = boardSize, nExtra = 4)
model.load_state_dict(torch.load("models/validMove"))
x = 0
y = 0
currentGame.simulateClick(x, y)
inputs = []
isValid = []
plusOne.libplusOne_python.Game.allIsValid(currentGame, inputs, isValid)

standardisedBoard = standardiseCategorical(currentGame, boardSize = boardSize)
[np.argmax(np.array(model(torch.Tensor(markMoveInCategorical(standardisedBoard, x = x, y = y).reshape(-1).tolist())).detach())) for x, y in inputs]
list(np.array(isValid).astype(int))
