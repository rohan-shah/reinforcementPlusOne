from tqdm import *
from game import *
import os
import torch
import torch.nn
import torch.optim
from network import plusOneStateChange

games = 10000

def standardiseData(game):
    return((game._board - game._max + 8) / 8)

loss_function = torch.nn.MSELoss()

predicter = plusOneStateChange()

if os.path.exists("models/stateChange"):
    predicter.load_state_dict(torch.load("models/stateChange"))
    print("Loading state change model")

optimizer = torch.optim.SGD(predicter.parameters(), lr=0.001, momentum=0.9)

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
            currentGame.click(action // 5, action % 5)
            trainingData["Input"].append(list(inputStandardised.reshape(-1)) + [action // 5, action % 5])
            trainingData["Target"].append(standardiseData(currentGame).reshape(-1))
            trainingData["Output"].append(predicter(torch.Tensor(list(standardiseData(currentGame).reshape(-1)) + [action // 5, action % 5])))
    
    optimizer.zero_grad()
    loss = loss_function(torch.Tensor(np.stack(trainingData["Target"])), torch.stack(trainingData["Output"]))
    loss.backward()
    optimizer.step()
    torch.save(predicter.state_dict(), "models/stateChange")
    print("Loss: {}".format(loss))
