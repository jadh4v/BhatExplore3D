cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(DistField CXX CUDA)
# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
 
find_package(Qt5 COMPONENTS Core Gui OpenGL Widgets REQUIRED)
find_package(Eigen3 REQUIRED)
#find_package(CUDA 11.2 EXACT REQUIRED)
#find_package(CUDA 11.2 EXACT REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS}) 
include_directories(${CUDA_INCLUDE_DIRS})

set(VOLDEF_SOURCES
    #AlgoExtractPolyLine.cpp
    #AlgoExtractPolyLine.h
    #AlgoImageExtractComponents.cpp
    #AlgoImageExtractComponents.h
    #AlgoImageConnectedComponents.cpp
    #AlgoImageConnectedComponents.h
    #AlgoTetrahedralMeshToSurface.cpp
    #AlgoTetrahedralMeshToSurface.h
    #AlgoCleanTetraMesh.cpp
    #AlgoCleanTetraMesh.h
    #AlgoControlCurveTransform.cpp
    #AlgoControlCurveTransform.h
    #AlgoVolumeCPR.cpp
    #AlgoVolumeCPR.h
    #AlgoVolumetricRegionToMesh.cpp
    #AlgoVolumetricRegionToMesh.h
    #AlgoImageToTetrahedra.cpp
    #AlgoImageToTetrahedra.h
    #AlgoDeformVolumeLSE.cpp
    #AlgoDeformVolumeLSE.h
    #AlgoVPR.cpp
    #AlgoVPR.h
    #AlgoVOI.cpp
    #AlgoVOI.h
    AlgoCudaDistanceField.cpp
    AlgoCudaDistanceField.h
    ComputeDistanceField.cu
    #CSRMatrix.cpp
    #CSRMatrix.h
    #CudaLinearSolver.cpp
    #CudaLinearSolver.h
    #pbicgstab.cpp
    #FImage.cpp
    #FImage.h
    #mmio.cpp
    #mmio.h
    #NvGraph_SSSP.cpp
    #Renderer.cpp
    #Renderer.h
    #VesselExtractionWindow.cpp
    #VesselExtractionWindow.h
)

#set(MY_CUDA_FILES
#ComputeDistanceField.cu)


# Let QT pre-process the files and add the generated files to the source list
qt5_wrap_ui(VOLDEF_GENERATED_SOURCES ${VOLDEF_UIS})
source_group("Generated Sources - Do Not Edit" FILES ${VOLDEF_GENERATED_SOURCES})

# Make sure the compiler can find the pre-processed files from qt4_wrap_ui
include_directories("${CMAKE_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/..")
include_directories("${PROJECT_SOURCE_DIR}/../../vp")
include_directories("${PROJECT_SOURCE_DIR}/../../vp/vpapp")
include_directories("${PROJECT_SOURCE_DIR}/../../vp/vpcommon")
include_directories("${PROJECT_SOURCE_DIR}/../../vp/cline")
include_directories("${PROJECT_SOURCE_DIR}/..")
include_directories("${PROJECT_SOURCE_DIR}/../vpcommon")
set (GLM_DIR "${PROJECT_SOURCE_DIR}/../../ThirdParty/glm-0.9.8.4")
#include_directories("C:/Program Files/eigen")
include_directories("C:/Program Files/Eigen3/include/eigen3")
include_directories( ${GLM_DIR} )
#include_directories("C:/ProgramData/NVIDIA Corporation/CUDA Samples/v11.2/common/inc")
include_directories("${PROJECT_SOURCE_DIR}/../../ThirdParty")

#cuda_add_library(DistField STATIC
#   ${MY_CUDA_FILES})
find_package(VTK REQUIRED) 
if(VTK_VERSION_MAJOR EQUAL 8 AND VTK_VERSION_MINOR LESS 9)
    include(${VTK_USE_FILE})
endif()

cuda_add_library(DistField STATIC  
    ${VOLDEF_SOURCES}
    ${VOLDEF_GENERATED_SOURCES} 
    ${VOLDEF_UIS} 
#    ${MY_CUDA_FILES}
)

target_link_libraries(DistField Qt5::Core Qt5::Widgets Qt5::OpenGL Qt5::Gui ${VTK_LIBRARIES}
#${ITK_LIBRARIES}
)

target_include_directories(DistField PUBLIC ${PROJECT_SOURCE_DIR})
