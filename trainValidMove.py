from tqdm import *
from game import *
import os
import torch
import torch.nn
import torch.optim
from network import plusOneValidMove

games = 10000

def standardiseData(game):
    return((game._board - game._max + 8) / 8)

loss_function = torch.nn.CrossEntropyLoss()

predicter = plusOneValidMove()

if os.path.exists("models/validMove"):
    predicter.load_state_dict(torch.load("models/validMove"))
    print("Loading valid move model")

optimizer = torch.optim.SGD(predicter.parameters(), lr=0.01, momentum=0.9)

counter = 0
while True:
    counter = counter + 1
    trainingData = {"Input": [], "Output": [], "Target": []}
    maximums = [0, 0]
    #We're going to play this number of games
    for gameIndex in tqdm(range(0, games), desc = "Run {}".format(counter)):
        currentGame = game()
        startingBoard = currentGame._board
        for action in range(0, 25):
            currentGame._board = startingBoard
            inputStandardised = standardiseData(currentGame)
            if currentGame.isValidMove(action // 5, action % 5):
                target = 1
            else:
                target = 0
            trainingData["Input"].append(list(inputStandardised.reshape(-1)) + [action // 5, action % 5])
            trainingData["Target"].append(target)
            trainingData["Output"].append(predicter(torch.Tensor(list(standardiseData(currentGame).reshape(-1)) + [action // 5, action % 5])))
    
    optimizer.zero_grad()
    loss = loss_function(torch.stack(trainingData["Output"]), torch.Tensor(np.stack(trainingData["Target"])).type(torch.LongTensor))
    loss.backward()
    optimizer.step()
    torch.save(predicter.state_dict(), "models/validMove")
    print("Loss: {}".format(loss))
