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

games = 100

loss_function = torch.nn.CrossEntropyLoss()

boardSize = 4

predicter = plusOneValidMoveAll(nLayers = 1, boardSize = boardSize, layerSize = boardSize*boardSize*9)

if os.path.exists("models/validMoveAll"):
    predicter.load_state_dict(torch.load("models/validMoveAll"))
    print("Loading valid move model")

optimizer = torch.optim.Adam(predicter.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x : 0.995 ** x)

for counter in range(100*10000):
    trainingData = {"Input": [], "Output": [], "Target": []}
    maximums = [0, 0]
    #We're going to play this number of games
    for gameIndex in range(0, games):#tqdm(range(0, games), desc = "Run {}".format(counter)):
        currentGame = plusOne.libplusOne_python.Game(boardSize)
        inputs = []
        isValid = []
        plusOne.libplusOne_python.Game.allIsValid(currentGame, inputs, isValid)
        isValid = np.array(isValid).astype(int)
        standardisedBoard = standardiseCategorical(currentGame, boardSize = boardSize)
        trainingData["Input"].append(standardisedBoard.reshape(-1))
        trainingData["Target"].append(isValid)
    
    inputs = torch.Tensor(np.stack(trainingData["Input"]))
    targets = torch.Tensor(np.stack(trainingData["Target"])).type(torch.LongTensor)
    output = predicter(inputs)
    optimizer.zero_grad()
    losses = [loss_function(output[:, x, :], targets[:, x]) for x in range(boardSize * boardSize)]
    total_losses = sum(losses)
    total_losses.backward()
    optimizer.step()
    if counter % 100 == 0:
        scheduler.step()
        print("Loss: {}".format(total_losses))
        torch.save(predicter.state_dict(), "models/validMoveAll")
