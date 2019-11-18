#include "pybind11_wrapper.h"
#include "game.h"

namespace py = pybind11;
namespace plusOne
{
	void simulateAllActions(Game* inputGame, py::list outputs)
	{
		int boardSize = inputGame->getBoardSize();
		for(int row = 0; row < boardSize; row++)
		{
			for(int column = 0; column < boardSize; column++)
			{
				std::pair<Game::boardType, int> newState = inputGame->simulateClick(column, row);
				outputs.append(py::make_tuple(column, row, newState.first, newState.second));
			}
		}
	}
	void allIsValid(Game* inputGame, py::list inputs, py::list isValid)
	{
		int boardSize = inputGame->getBoardSize();
		for(int row = 0; row < boardSize; row++)
		{
			for(int column = 0; column < boardSize; column++)
			{
				inputs.append(py::make_tuple(column, row));
				isValid.append(inputGame->isValidMove(column, row));
			}
		}
	}
}

PYBIND11_MODULE(libplusOne_python, m) {
    {
        py::class_<plusOne::Game>(m, "Game")
		.def(py::init<int>())
		.def("click", &plusOne::Game::click)
		.def("getBoard", &plusOne::Game::getBoard)
		.def("isValidMove", &plusOne::Game::isValidMove)
		.def("getMax", &plusOne::Game::getMax)
		.def("setBoard", &plusOne::Game::setBoard)
		.def("setMax", &plusOne::Game::setMax)
		.def("simulateClick", &plusOne::Game::simulateClick)
        	.def_static("simulateAllActions", &plusOne::simulateAllActions)
        	.def_static("allIsValid", &plusOne::allIsValid);
    }
}
