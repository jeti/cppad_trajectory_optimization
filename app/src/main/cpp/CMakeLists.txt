cmake_minimum_required(VERSION 3.5)

project(tester)

set(CMAKE_CXX_STANDARD 11)

# Location of header files
include_directories(${CMAKE_SOURCE_DIR}/../../../libs/include
        ${CMAKE_SOURCE_DIR}/../../../libs/include/coin
        ${CMAKE_SOURCE_DIR}/../../../libs/include/coin/ThirdParty)

add_executable(tester test_constraints.cpp)