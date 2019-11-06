from tqdm import *
from game import *
import os
import torch
import torch.nn
import torch.optim
from network import plusOnePlayer

discount = 0.9
steps = 50
games = 10000
epsilon = 0.1

reward_invalid_move = -10
reward_next_level = 100
reward_valid_move = 0.1
reward_failure = -1000
keep_proportion = 0.1

def standardiseData(game):
    return((game._board - game._max + 8) / 8)

loss_function = torch.nn.MSELoss()

model1 = plusOnePlayer()
model2 = plusOnePlayer()
models = [model1, model2]

if os.path.exists("models/model1"):
    model1.load_state_dict(torch.load("models/model1"))
    print("Loading model 1")
if os.path.exists("models/model2"):
    model2.load_state_dict(torch.load("models/model2"))
    print("Loading model 2")

optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.001, momentum=0.9)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.001, momentum=0.9)

counter = 0
while True:
    counter = counter + 1
    trainingData = [{"Prediction": [], "Target": []}, {"Prediction": [], "Target": []}]
    maximums = [0, 0]
    #We're going to play this number of games
    for gameIndex in tqdm(range(0, games), desc = "Run {}".format(counter)):
        #We have two models, training each other
        for modelIndex in range(0, 2):
            model = models[modelIndex]
            otherModel = models[1 - modelIndex]
            currentGame = game()
            for stepIndex in range(0, steps):
                startingState = standardiseData(currentGame)
                #Choose a random action
                qMatrix = model(torch.Tensor([startingState]))
                if np.random.uniform() < epsilon:
                    action = np.random.randint(low = 0, high = 25)
                else:
                    action = torch.argmax(qMatrix)
                result = currentGame.click(int(action / 5), action % 5)
                endingState = standardiseData(currentGame)
                failed = currentGame.isFailed()
                if failed:
                    bestExpectedReward = reward_failure
                else:
                    with torch.no_grad():
                        bestExpectedReward = torch.max(otherModel(torch.Tensor([endingState])))
                if np.random.uniform() < keep_proportion:
                    trainingData[modelIndex]["Prediction"].append(qMatrix[0, action])
                    if result == -1:
                        trainingData[modelIndex]["Target"].append(reward_invalid_move + discount * bestExpectedReward)
                    elif result == 0:
                        trainingData[modelIndex]["Target"].append(reward_valid_move + discount * bestExpectedReward)
                    else:
                        maximums[modelIndex] = max(maximums[modelIndex], currentGame._max)
                        trainingData[modelIndex]["Target"].append(reward_next_level + discount * bestExpectedReward)
                if failed:
                    break
    
    optimizer1.zero_grad()
    loss1 = loss_function(torch.stack(trainingData[0]["Prediction"]), torch.Tensor(trainingData[0]["Target"]))
    loss1.backward()
    optimizer1.step()

    optimizer2.zero_grad()
    loss2 = loss_function(torch.stack(trainingData[1]["Prediction"]), torch.Tensor(trainingData[1]["Target"]))
    loss2.backward()
    optimizer2.step()
    print("Loss1: {}, Loss2: {}".format(loss1, loss2))
    print("Max1: {}, Max2: {}".format(maximums[0], maximums[1]))

    torch.save(model1.state_dict(), "models/model1")
    torch.save(model2.state_dict(), "models/model2")

