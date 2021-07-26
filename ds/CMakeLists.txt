cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

# Set package properties
#set(CMAKE_CXX_FLAGS "/MT /EHsc")
project(DS CXX)
 
# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

###
# Perform build configuration of dependencies
# Set up QT5 and required components
find_package(Qt5 COMPONENTS Core Gui REQUIRED)
find_package(VTK REQUIRED)
if(VTK_VERSION_MAJOR EQUAL 8 AND VTK_VERSION_MINOR LESS 9)
    include(${VTK_USE_FILE})
endif()

set(DS_SOURCES
    AlgoImageToVtk.cpp
    AlgoImageToVtk.h
    AlgoVtkToImage.cpp
    AlgoVtkToImage.h
    BitImage.cpp
    BitImage.h
    Bitvector.h
    BoundingBox.cpp
    BoundingBox.h
    codebase.h
    CompressedSegment.cpp
    CompressedSegment.h
    CompressedSegmentation.cpp
    CompressedSegmentation.h
    DSObject.h
    Graph.cpp 
    Graph.h
    Graph_Edge.hpp  
    Graph_Node.cpp
    Graph_Node.h
    Grid.cpp
    Grid.h
    GridCell.cpp
    GridCell.h
    GridPoint.cpp
    GridPoint.h
    Image.cpp
    Image.h
    ImageSegmentation.cpp
    ImageSegmentation.h
    Histogram.h
    ParametricPolyline.cpp
    ParametricPolyline.h
    Renamer.cpp
    Renamer.h
    ShallowTree.cpp
    ShallowTree.h
    ShallowTree_node.hpp
    VoxelRow.cpp
    VoxelRow.h
)

#source_group("Generated Sources - Do Not Edit" FILES ${DS_GENERATED_SOURCES})

# Make sure the compiler can find the pre-processed files from qt4_wrap_ui
include_directories(${PROJECT_SOURCE_DIR}/..)
add_library(DS STATIC ${DS_SOURCES})
target_link_libraries(DS Qt5::Core Qt5::Gui ${VTK_LIBRARIES})
target_include_directories(DS PUBLIC ${PROJECT_SOURCE_DIR})
