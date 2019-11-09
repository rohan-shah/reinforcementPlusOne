#pragma once
#include <Eigen/Eigen>
#include <random>
namespace plusOne
{
	class Game
	{
	public:
		Game();
		int click(int x, int y);
		void getIndicesForClick(int x, int y);
		static std::mt19937 randomSource;
		static std::vector<std::pair<int, int> > indices;
		struct searchState
		{
		public:
			searchState(int x, int y, int direction)
				:x(x), y(y), direction(direction)
			{}
			int x, y, direction;
		};
		static std::vector<searchState> dfsStates;
		typedef Eigen::Matrix<int, 5, 5> boardType;
		boardType getBoard();
		std::pair<boardType, int> simulateClick(int x, int y);
		bool isValidMove(int x, int y);
		int getMax();
		void setBoard(boardType board);
		void setMax(int max);
	private:
		static void fill(boardType& board, int min, int max);
		static void shiftDown(boardType& board);
		void incrementRange();
		boardType board;
		typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> maskType;
		maskType mask;
		int min, max;
	};
}
