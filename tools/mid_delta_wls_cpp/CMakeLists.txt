cmake_minimum_required(VERSION 3.20)
project(mid_delta_wls_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(
    mid_delta_wls_cpp
    src/main.cpp
    src/batch_loader.cpp
    src/csv_reader.cpp
    src/wls_regression.cpp
    src/grouping.cpp
    ../../decoupling/decouplingEngine.cpp
)

target_include_directories(
    mid_delta_wls_cpp
    PRIVATE
        src
        ../../decoupling
)

if(MSVC)
    target_compile_options(mid_delta_wls_cpp PRIVATE /W4 /permissive-)
else()
    target_compile_options(mid_delta_wls_cpp PRIVATE -Wall -Wextra -Wpedantic)
endif()
