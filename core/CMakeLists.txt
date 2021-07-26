cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(Core CXX)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

set(CORE_SOURCES
    Algorithm.cpp
    Algorithm.h
    macros.h
    CoreTypedefs.h
)

add_library(Core STATIC ${CORE_SOURCES} )
target_link_libraries(Core)
target_include_directories(Core PUBLIC ${PROJECT_SOURCE_DIR})
