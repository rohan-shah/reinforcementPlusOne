from tqdm import *
import plusOne
import os
import torch
import torch.nn
import torch.optim
import torch.optim.lr_scheduler
from plusOne.networkCategorical import plusOneValidMoveAll, standardiseCategorical, standardiseCategoricalSimulated, markMoveInCategorical
#from network import plusOneValidMove, standardiseData
import numpy as np

boardSize = 4

predicter = plusOneValidMoveAll(nLayers = 1, boardSize = boardSize, layerSize = boardSize*boardSize*9)

predicter.load_state_dict(torch.load("models/validMoveAll"))
print("Loading valid move model")

for k in range(12):
    currentGame = plusOne.libplusOne_python.Game(boardSize)
standardisedBoard = standardiseCategorical(currentGame, boardSize = boardSize)
inputs = torch.Tensor(standardisedBoard.reshape([1, 4*4*9]))
result = predicter(inputs)
result = result.detach().numpy()[0]
isValidPrediction = np.array([(value, x // 4, x% 4) for x, value, in enumerate(result.argmax(1))])
isValidPrediction

board = currentGame.getBoard()
board

copiedBoard = board.copy()
for valid, x, y, in isValidPrediction:
    if valid == 1:
        copiedBoard[x, y] = 0

copiedBoard
