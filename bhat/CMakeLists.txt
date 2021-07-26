cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

# Set package properties
project(Bhat CXX CUDA)
 
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
find_package(Eigen3 REQUIRED)
include_directories ( ${EIGEN3_INCLUDE_DIRS} )


set(BHAT_SOURCES
    AbstractRegion.cpp
    AbstractRegion.h
    AttributeGenerator.cpp
    AttributeGenerator.h
    CudaTexture.cu
    CudaTexture.h
    FeatureSpace.cpp
    FeatureSpace.h
    NGrid.cpp
    NGrid.h
    FullRegion.cpp
    FullRegion.h
    Region.cpp
    Region.h
    FeatureSpaceCuda.cu
    FeatureSpaceCuda.h
    FeatureSpaceCudaFunctors.h
    FeatureSpaceCudaKernels.h
    Utils.h
)

#source_group("Generated Sources - Do Not Edit" FILES ${DS_GENERATED_SOURCES})
# Make sure the compiler can find the pre-processed files from qt4_wrap_ui
#include(${VTK_USE_FILE})
include_directories(${PROJECT_SOURCE_DIR}/..)
include_directories("C:/Program\ Files/Eigen3/include/eigen3")
cuda_add_library(Bhat STATIC ${BHAT_SOURCES})
target_link_libraries(Bhat Qt5::Core Qt5::Gui ${VTK_LIBRARIES})
target_include_directories(Bhat PUBLIC ${PROJECT_SOURCE_DIR})
