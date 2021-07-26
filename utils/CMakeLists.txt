cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(Utils CXX)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5 COMPONENTS Core Gui REQUIRED)
find_package(VTK REQUIRED)
if(VTK_VERSION_MAJOR EQUAL 8 AND VTK_VERSION_MINOR LESS 9)
    include(${VTK_USE_FILE})
endif()
#find_package(ITK REQUIRED)
#include(${ITK_USE_FILE})

set(UTILS_SOURCES
	utils.cpp
	utils.h
)

include_directories("${CMAKE_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/../../fvis")
include_directories("${PROJECT_SOURCE_DIR}/../../vp")
include_directories("${PROJECT_SOURCE_DIR}/../../vp/vpcommon")
include_directories("${PROJECT_SOURCE_DIR}/../../vp/cline")

#source_group("Generated Sources - Do Not Edit" FILES ${DS_GENERATED_SOURCES})

# Make sure the compiler can find the pre-processed files from qt4_wrap_ui
include_directories(${PROJECT_SOURCE_DIR}/..)
add_library(Utils STATIC ${UTILS_SOURCES})
target_link_libraries(Utils DS IO Qt5::Core Qt5::Gui ${VTK_LIBRARIES} #${ITK_LIBRARIES}
    )
target_include_directories(Utils PUBLIC ${PROJECT_SOURCE_DIR})
