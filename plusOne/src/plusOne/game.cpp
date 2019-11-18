#include "game.h"
#include <iostream>
namespace plusOne
{
	std::mt19937 Game::randomSource;
	std::vector<std::pair<int, int> > Game::indices;
	std::vector<Game::searchState> Game::dfsStates;
	inline void intToDirection(int& x, int& y, int direction)
	{
		switch(direction)
		{
		case 0:
			x = 1; y = 0;
			break;
		case 1:
			x = 0; y = 1;
			break;
		case 2:
			x = -1, y = 0;
			break;
		case 3:
			x = 0; y = -1;
			break;
		}
	}
	Game::Game(int boardSize)
		: boardSize(boardSize), min(1), max(7), board(boardSize, boardSize)
	{
		std::uniform_int_distribution<> dist(min, max - 2);
		for(int i = 0; i < boardSize; i++)
		{
			for(int j = 0; j < boardSize; j++)
			{
				board(i, j) = dist(randomSource);
			}
		}
		std::uniform_int_distribution<> positionDist(0, boardSize - 1);
		int x = positionDist(randomSource);
		int y = positionDist(randomSource);
		board(y, x) = max - 1;
	}
	void Game::getIndicesForClick(int x, int y)
	{
		int currentValue = board(y, x);
		indices.clear();
		dfsStates.clear();

		mask = maskType::Constant(boardSize, boardSize, false);
		mask(y, x) = true;

		indices.push_back(std::make_pair(x, y));

		dfsStates.push_back(searchState(x, y, 0));
		while(dfsStates.size() > 0)
		{
			searchState& current = *dfsStates.rbegin();
			if(current.direction == 4)
			{
				dfsStates.pop_back();
			}
			else
			{
				int directionX, directionY;
				intToDirection(directionX, directionY, current.direction);
				current.direction++;
				int newX = current.x + directionX;
				int newY = current.y + directionY;
				if(newX < 0 || newY < 0 || newX >= boardSize || newY >= boardSize || mask(newY, newX) || board(newY, newX) != currentValue)
				{
					continue;
				}
				else
				{
					mask(newY, newX) = true;
					indices.push_back(std::make_pair(newX, newY));
					dfsStates.push_back(searchState(newX, newY, 0));
				}
			}
		}
	}
	void Game::incrementRange()
	{
		max++;
		min = std::max(1, max - 8);
	}
	void Game::shiftDown(Game::boardType& board, int boardSize)
	{
		for(int column = 0; column < boardSize; column++)
		{
			int row = boardSize - 1;
			bool allNegativeOne = true;
			for(int i = 0; i < row; i++) allNegativeOne &= board(i, column) == -1;
			while(row > 0 and !allNegativeOne)
			{
				if(board(row, column) == -1)
				{
					for(int i = row; i > 0; i--)
					{
						board(i, column) = board(i - 1, column);
					}
					board(0, column) = -1;
				}
				else row--;
				allNegativeOne = true;
				for(int i = 0; i < row; i++) allNegativeOne &= board(i, column) == -1;
			}
		}
	}
	void Game::fill(Game::boardType& board, int min, int max, int boardSize)
	{
		std::uniform_int_distribution<> dist(min, max - 2);
		for(int i = 0; i < boardSize; i++)
		{
			for(int j = 0; j < boardSize; j++)
			{
				if(board(i, j) == -1) board(i, j) = dist(randomSource);
			}
		}
	}
	std::pair<Game::boardType, int> Game::simulateClick(int x, int y)
	{
		if(x < 0 || y < 0 || x >= boardSize || y >= boardSize)
		{
			return std::make_pair(board, max);
		}
		int currentValue = board(y, x);
		Game::getIndicesForClick(x, y);
		if(indices.size() > 1)
		{
			boardType copiedBoard = board;
			for(std::pair<int, int> p : indices)
			{
				copiedBoard(p.second, p.first) = -1;
			}
			bool nextLevel;
			if(currentValue == max - 1)
			{
				nextLevel = true;
				int currentMin = copiedBoard.minCoeff();
				for(int i = 0; i < boardSize; i++)
				{
					for(int j = 0; j < boardSize; j++)
					{
						if(copiedBoard(i, j) == currentMin) copiedBoard(i, j) = -1;
					}
				}
			}
			else
			{
				nextLevel = false;
			}
			copiedBoard(y, x) = currentValue + 1;
			shiftDown(copiedBoard, boardSize);
			fill(copiedBoard, min + nextLevel, max + nextLevel, boardSize);
			if(nextLevel) return std::make_pair(copiedBoard, max + 1);
			return std::make_pair(copiedBoard, max);
		}
		return std::make_pair(board, max);
	}
	int Game::click(int x, int y)
	{
		if(x < 0 || y < 0 || x >= boardSize || y >= boardSize)
		{
			return -1;
		}
		int currentValue = board(y, x);
		Game::getIndicesForClick(x, y);
		if(indices.size() > 1)
		{
			for(std::pair<int, int> p : indices)
			{
				board(p.second, p.first) = -1;
			}
			bool nextLevel;
			if(currentValue == max - 1)
			{
				nextLevel = true;
				int currentMin = board.minCoeff();
				for(int i = 0; i < boardSize; i++)
				{
					for(int j = 0; j < boardSize; j++)
					{
						if(board(i, j) == currentMin) board(i, j) = -1;
					}
				}
				incrementRange();
			}
			else
			{
				nextLevel = false;
			}
			board(y, x) = currentValue + 1;
			shiftDown(board, boardSize);
			fill(board, min, max, boardSize);
			if(nextLevel) return 1;
			return 0;
		}
		return -1;
	}
	Game::boardType Game::getBoard()
	{
		return board;
	}
	bool Game::isValidMove(int x, int y)
	{
		int currentValue = board(y, x);
		for(int direction = 0; direction < 4; direction++)
		{
			int directionX, directionY;
			intToDirection(directionX, directionY, direction);
			int newX = x + directionX;
			int newY = y + directionY;
			if(newX < 0 || newY < 0 || newX >= boardSize || newY >= boardSize)
			{
				continue;
			}
			if(board(newY, newX) == currentValue) return true;
		}
		return false;
	}
	int Game::getMax()
	{
		return max;
	}
	void Game::setMax(int max)
	{
		max = max;
		min = std::max(1, max - 8);
	}
	void Game::setBoard(Game::boardType board)
	{
		this->board = board;
	}
	int Game::getBoardSize() const
	{
		return boardSize;
	}
}
