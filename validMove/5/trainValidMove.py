from tqdm import *
import plusOne
import os
import torch
import torch.nn
import torch.optim
import torch.optim.lr_scheduler
from plusOne.networkCategorical import plusOneValidMove, standardiseCategorical, standardiseCategoricalSimulated, markMoveInCategorical
#from network import plusOneValidMove, standardiseData
import numpy as np

games = 50000

loss_function = torch.nn.CrossEntropyLoss()

boardSize = 5

predicter = plusOneValidMove(nLayers = 7, boardSize = boardSize, nExtra = 4)

if os.path.exists("models/validMove"):
    predicter.load_state_dict(torch.load("models/validMove"))
    print("Loading valid move model")

optimizer = torch.optim.Adam(predicter.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x : 0.995 ** x)

counter = 0
while True:
    counter = counter + 1
    trainingData = {"Input": [], "Output": [], "Target": []}
    maximums = [0, 0]
    #We're going to play this number of games
    for gameIndex in tqdm(range(0, games), desc = "Run {}".format(counter)):
        currentGame = plusOne.libplusOne_python.Game(boardSize)
        inputs = []
        isValid = []
        plusOne.libplusOne_python.Game.allIsValid(currentGame, inputs, isValid)
        isValid = np.array(isValid).astype(int)
        standardisedBoard = standardiseCategorical(currentGame, boardSize = boardSize)
        inputs = [list(markMoveInCategorical(standardisedBoard, x = x, y = y).reshape(-1)) for (x, y) in inputs]
        trainingData["Input"].append(inputs)
        trainingData["Target"].append(isValid)
    
    target = torch.Tensor(np.concatenate(trainingData["Target"])).type(torch.LongTensor)
    inputs = torch.Tensor(np.concatenate(trainingData["Input"])).type(torch.FloatTensor)
    for counter in range(50):
        output = predicter(inputs)
        optimizer.zero_grad()
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        print("Loss: {}".format(loss))
    torch.save(predicter.state_dict(), "models/validMove")
