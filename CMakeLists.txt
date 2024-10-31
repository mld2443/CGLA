cmake_minimum_required(VERSION 3.27)
project(CGLA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(${PROJECT_NAME} linearAlgebra.cpp)

add_executable(playground playground.cpp)
