cmake_minimum_required(VERSION 4.0)

project(CGLA LANGUAGES CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -Wconversion -Wsign-conversion
        -Wno-missing-field-initializers
    )
elseif(MSVC)
    add_compile_options(/W4 /permissive-)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

add_executable(${PROJECT_NAME})

target_sources(${PROJECT_NAME}
    PRIVATE
        src/main.cpp

    PRIVATE FILE_SET modules TYPE CXX_MODULES FILES
        src/util/common.cppm
        src/util/meta.cppm
        src/linalg/reference.cppm
        src/linalg/pointer.cppm
        src/linalg/value.cppm
        src/linalg/tensor.cppm
)
