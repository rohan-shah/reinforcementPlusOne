import numpy as np
class game:
    def __init__(self):
        self._board = np.random.randint(low = 1, high = 7, size = (5, 5)) 
        self._min = 1
        self._max = 6
        self._mask = np.array([False] * 25).reshape((5, 5))

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
                if newX < 0 or newX > 4 or newY < 0 or newY > 4 or self._mask[newY, newX] or self._board[newY, newX] != currentValue:
                    next
                else:
                    self._mask[newY, newX] = True
                    indices.append((newX, newY))
                    state.append((newX, newY, iter(directions)))
        return indices

    def click(self, x, y):
        x = int(x)
        y = int(y)
        if x < 0 or y < 0 or x >= 5 or y >= 5:
            return(-1)
        else:
            currentValue = self._board[y, x]
            indices = self.getIndicesForClick(x, y)
            if len(indices) > 1:
                for (x, y) in indices:
                    self._board[y, x] = -1
                if currentValue == self._max:
                    currentMin = np.min(self._board)
                    for row in range(0, 5):
                        for column in range(0, 5):
                            if self._board[row, column] == currentMin:
                                self._board[row, column] = -1
                    self.increment_range()
                import pdb; pdb.set_trace()
                self.shift_down()
                self.fill()
                return(1)
            return(0)

    def increment_range(self):
        self._max = self._max + 1
        self._min = min(1, self._max - 8)

    def shift_down(self):
        for column in range(0, 5):
            row = 4
            while row > 0 and ~np.all(self._board[:row, column] == -1):
                if self._board[row, column] == -1:
                    self._board[range(1, row+1), column] = self._board[range(0, row), column]
                    self._board[0, column] = -1
                else:
                    row = row - 1

    def fill(self):
        for column in range(0, 5):
            for row in range(0, 5):
                if self._board[row, column] == -1:
                    self._board[row, column] = np.random.randint(low = self._min, high = self._max+1)
