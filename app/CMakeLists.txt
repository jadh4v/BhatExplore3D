cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(Bhattacharyya LANGUAGES CXX CUDA)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed. This is for Qt.
set(CMAKE_AUTOMOC ON)
#set(CMAKE_CUDA_FLAGS -gencode=arch=compute_70,code=\"sm_70,compute_70\"   -gencode=arch=compute_50,code=sm_50)
#set(CMAKE_CUDA_FLAGS -gencode=arch=compute_70,code=sm_70,compute_70)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --expt-extended-lambda --extended-lambda)
set(CMAKE_CUDA_FLAGS -gencode=arch=compute_75,code=sm_75)
message(${CMAKE_CUDA_FLAGS})

find_package(Qt5 COMPONENTS Core Gui OpenGL Widgets Quick QuickWidgets REQUIRED)

find_package(VTK REQUIRED)
if(VTK_VERSION_MAJOR EQUAL 8 AND VTK_VERSION_MINOR LESS 9)
    include(${VTK_USE_FILE})
endif()
find_package(OpenGL REQUIRED) 
include_directories( ${OPENGL_INCLUDE_DIRS} )
find_package(glew REQUIRED)
include_directories(${GLEW_INCLUDE_DIR})


#find_package(CUDA 11.2 EXACT REQUIRED) 
find_package(CUDA 10.2 EXACT REQUIRED) 
include_directories(${CUDA_INCLUDE_DIRS})

find_package(eigen3 REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS} )
include_directories("C:/Program\ Files/Eigen3/include/eigen3")
include_directories( ${EIGEN3_INCLUDE_DIRS} )

set(ThirdParty_DIR "${PROJECT_SOURCE_DIR}/../../ThirdParty")
set(ITK_DIR "C:/gitlab/build/itk-5.0.1")
find_package(ITK REQUIRED)
include(${ITK_USE_FILE}) 

set (BHATT_SOURCES
     ActiveContourBlock.cpp
     ActiveContourBlock.cu
     ActiveContourBlock.h
     bhatt_global.h
     BhattParameters.cpp
     BhattParameters.h
     TreeView.h
     TreeView.cpp
     CudaFunctions.cu
     CudaFunctions.h
     GreyScaleWidget.cpp
     GreyScaleWidget.h
     HistogramViewer.cpp
     HistogramViewer.h
     main.cpp
     #main_cmd.cpp
     #main_cpu.cpp
     MultiBlockSync.cpp
     MultiBlockSync.h
     ProjUtils.cpp
     ProjUtils.h
     ROIBoxCallback.h 
     RenderCallback.h 
     Slider.cpp
     Slider.h
     ThreadSyncObject.cpp
     ThreadSyncObject.h
     UserInterface.cpp 
     UserInterface.h 
     VolumeVisualizer.cpp
     VolumeVisualizer.h
     ../GenericExplorer/Bookmark.cpp 
     ../GenericExplorer/Bookmark.h
     ../GenericExplorer/BookmarkListModel.cpp 
     ../GenericExplorer/BookmarkListModel.h
     ../GenericExplorer/BookmarkTool.cpp 
     ../GenericExplorer/BookmarkTool.h
     ../GenericExplorer/DialogOpticalProperties.cpp 
     ../GenericExplorer/DialogOpticalProperties.h
     ../GenericExplorer/VolumeStats.h
     ../GenericExplorer/VolumeStats.cpp
)

set (BHATT_UIS roi_clipping_box.ui)

#file(GLOB BHATT_SOURCES *.cpp *.h)
#message(${BHATT_SOURCES})
  
# Let QT pre-process the files and add the generated files to the source list
qt5_wrap_ui(GENERATED_SOURCES ${BHATT_UIS})
source_group("Generated Sources - Do Not Edit" FILES ${GENERATED_SOURCES})

# Make sure the compiler can find the pre-processed files from qt4_wrap_ui
include_directories("${CMAKE_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/..")
include_directories(${ThirdParty_DIR})
include_directories("${ThirdParty_DIR}/glm-0.9.8.4")
#include_directories("C:/ProgramData/NVIDIA Corporation/CUDA Samples/v11.2/common/inc")
include_directories("C:/ProgramData/NVIDIA Corporation/CUDA Samples/v10.2/common/inc")
add_subdirectory(${PROJECT_SOURCE_DIR}/../core libCore)
add_subdirectory(${PROJECT_SOURCE_DIR}/../ds libDS)
add_subdirectory(${PROJECT_SOURCE_DIR}/../io libIO)
add_subdirectory(${PROJECT_SOURCE_DIR}/../utils libUtils)
add_subdirectory(${PROJECT_SOURCE_DIR}/../DistField libDistField)
add_subdirectory(${PROJECT_SOURCE_DIR}/../bhat libBhat)
#add_subdirectory(${PROJECT_SOURCE_DIR}/../graphseg libGraphSeg)
add_subdirectory(${PROJECT_SOURCE_DIR}/../graphics libGraphics)
add_subdirectory(${PROJECT_SOURCE_DIR}/../vpcommon vpcommon)

set(ANACONDA_DIR "c:/Anaconda3/Library/lib")

#MATLAB
set(MATLAB_ROOT "D:/Program Files/MATLAB/R2020a/extern")
include_directories("${MATLAB_ROOT}/include")
link_directories("${MATLAB_ROOT}/lib/win64/microsoft")

#ZLIB
find_package(zlib REQUIRED)

#LIBICS
set(LIBICS_CPP_DIR "C:/Program\ Files/libics/cmake")
include_directories("${LIBICS_CPP_DIR}/../include")

cuda_add_executable(Bhattacharyya  
     ${BHATT_SOURCES}
     ${GENERATED_SOURCES}
     ${BHATT_UIS}
     cpp.hint
    )

target_link_libraries(Bhattacharyya
	Core DS Bhat IO vpcommon Utils #GraphSeg
    Graphics DistField
    ${GLEW_LIBRARIES} Qt5::Core Qt5::Widgets Qt5::OpenGL Qt5::Gui Qt5::Quick Qt5::QuickWidgets
    ${VTK_LIBRARIES} 
    ${ITK_LIBRARIES} 
    ${OPENGL_LIBRARIES} 
    ${ANACONDA_DIR}/icu*.lib
    debug ${LIBICS_CPP_DIR}/../lib/Debug/libics.lib
    debug ${LIBICS_CPP_DIR}/../lib/Debug/libics_cpp.lib
    optimized ${LIBICS_CPP_DIR}/../lib/Release/libics.lib
    optimized ${LIBICS_CPP_DIR}/../lib/Release/libics_cpp.lib
    ${ANACONDA_DIR}/zlib.lib
    #libMatlabEngine.lib libMatlabDataArray.lib
    )
