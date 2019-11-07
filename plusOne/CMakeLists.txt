# ====================================================================
# Specify the minimum version for CMake
cmake_minimum_required(VERSION 3.5)

set(Boost_USE_STATIC_LIBS   OFF)

# ====================================================================
# We have found that `ninja` is much faster than `make`
set(CMAKE_GENERATOR "Ninja")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

# ====================================================================
# The project must be defined after selecting the compiler
project(plusOne)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(PythonInterp)
find_package(PythonLibs)

if(NOT WIN32)
    # openms has this annoying clang pragma that causes warnings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
endif()

# ====================================================================
# find the external packages

add_definitions(-DUSE_PYTHON3)

message(STATUS "PYTHON_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}")
message(STATUS "PYTHON_LIBRARY: ${PYTHON_LIBRARY}")
message(STATUS "PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_package(Threads REQUIRED)

if(WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS On)
endif()

# ====================================================================
# add the src and testing directories
add_subdirectory(pybind11)
add_subdirectory(src)
