import numpy as np
class game:
    def __init__(self, boardSize):
        self._boardSize = boardSize
        self._board = np.random.randint(low = 1, high = 7, size = (boardSize, boardSize)) 
        self._min = 1
        self._max = 6
        self._mask = np.array([False] * boardSize * boardSize).reshape((boardSize, boardSize))

    def getIndicesForClick(self, x, y):
        currentValue = self._board[y, x]
        self._mask[:, :] = False
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        state = [(x, y, iter(directions))]
        self._mask[y, x] = True
        indices = [(x, y)]
        while len(state) > 0:
            x, y, direction = state[-1]
            nextDirection = next(direction, None)
            if nextDirection is None:
                state.pop()
            else:
                newX = x + nextDirection[0]
                newY = y + nextDirection[1]
                if newX < 0 or newX > self._boardSize - 1 or newY < 0 or newY > self._boardSize - 1 or self._mask[newY, newX] or self._board[newY, newX] != currentValue:
                    next
                else:
                    self._mask[newY, newX] = True
                    indices.append((newX, newY))
                    state.append((newX, newY, iter(directions)))
        return indices

    def click(self, x, y):
        x = int(x)
        y = int(y)
        if x < 0 or y < 0 or x >= self._boardSize or y >= self._boardSize:
            return(-1)
        else:
            currentValue = self._board[y, x]
            indices = self.getIndicesForClick(x, y)
            if len(indices) > 1:
                for (x, y) in indices:
                    self._board[y, x] = -1
                if currentValue == self._max:
                    nextLevel = True
                    currentMin = np.min(self._board)
                    for row in range(0, self._boardSize):
                        for column in range(0, self._boardSize):
                            if self._board[row, column] == currentMin:
                                self._board[row, column] = -1
                    self.increment_range()
                else:
                    nextLevel = False
                self._board[y, x] = currentValue + 1
                self.shift_down()
                self.fill()
                if nextLevel:
                    return(1)
                else:
                    return(0)
            return(-1)

    def increment_range(self):
        self._max = self._max + 1
        self._min = max(1, self._max - 8)

    def shift_down(self):
        for column in range(0, self._boardSize):
            row = self._boardSize - 1
            while row > 0 and ~np.all(self._board[:row, column] == -1):
                if self._board[row, column] == -1:
                    self._board[range(1, row+1), column] = self._board[range(0, row), column]
                    self._board[0, column] = -1
                else:
                    row = row - 1

    def fill(self):
        for column in range(0, self._boardSize):
            for row in range(0, self._boardSize):
                if self._board[row, column] == -1:
                    self._board[row, column] = np.random.randint(low = self._min, high = self._max)
    
    def isValidMove(self, x, y):
        if x < 0 or y < 0 or x >= self._boardSize or y >= self._boardSize:
            return(False)
        currentValue = self._board[y, x]
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        targets = [(direction[0] + y, direction[1] + x) for direction in directions]
        targets = [(y, x) for (y, x) in targets if not (x < 0 or y < 0 or x >= self._boardSize or y >= self._boardSize)]
        values = [self._board[y, x] for (y, x) in targets]
        return(currentValue in values)

    def isFailed(self):
        if np.all(self._board[1:, :] != self._board[:-1, :]) and np.all(self._board[:, 1:] != self._board[:, :-1]):
            return(True)
        return(False)
