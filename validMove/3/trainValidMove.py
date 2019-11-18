from tqdm import *
import plusOne
import os
import torch
import torch.nn
import torch.optim
import torch.optim.lr_scheduler
from networkCategorical import plusOneValidMove, standardiseCategorical, standardiseCategoricalSimulated, markMoveInCategorical
#from network import plusOneValidMove, standardiseData
import numpy as np

games = 20000

loss_function = torch.nn.CrossEntropyLoss()

boardSize = 3

predicter = plusOneValidMove(nLayers = 8, boardSize = boardSize)

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
        trainingData["Output"].append(predicter(torch.Tensor(trainingData["Input"][-1])))
    
    optimizer.zero_grad()
    loss = loss_function(torch.cat(trainingData["Output"]), torch.Tensor(np.concatenate(trainingData["Target"])).type(torch.LongTensor))
    loss.backward()
    optimizer.step()
    scheduler.step()
    torch.save(predicter.state_dict(), "models/validMove")
    print("Loss: {}".format(loss))
