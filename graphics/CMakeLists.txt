cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(Graphics CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt5 COMPONENTS Core Gui OpenGL Widgets REQUIRED)
# Let QT pre-process the files and add the generated files to the source list
#qt5_wrap_ui(GENERATED_SOURCES ${GRAPHICS_UIS})
source_group("Generated Sources - Do Not Edit" FILES ${GENERATED_SOURCES}) 

# Make sure the compiler can find the pre-processed files from qt4_wrap_ui
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories("${CMAKE_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/..")
include_directories("${PROJECT_SOURCE_DIR}/../fvis")
#include_directories("${PROJECT_SOURCE_DIR}/../../vp")
include_directories("${PROJECT_SOURCE_DIR}/../vpcommon")
set (GLM_DIR "${PROJECT_SOURCE_DIR}/../../ThirdParty/glm-0.9.8.4")
include_directories( ${GLM_DIR} )

find_package(OpenGL REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS} )
find_package(glew REQUIRED)
include_directories( ${GLEW_INCLUDE_DIR} )
find_package(VTK REQUIRED)

set(GRAPHICS_SOURCES
    AxialView.cpp
    AxialView.h
    ContinuousSliceView.cpp
    ContinuousSliceView.h 
    TextureContext.cpp 
    TextureContext.h 
    VolumeRenderer.cpp 
    VolumeRenderer.h 
    #DialogOpticalProperties.cpp 
    #DialogOpticalProperties.h 
)

set(GRAPHICS_SHADERS
    cube.vert.glsl 
    cube.frag.glsl 
    slice.vert.glsl 
    slice.frag.glsl 
    volume_rendering.vert.glsl 
    volume_rendering.frag.glsl 
)

#message(${GRAPHICS_SOURCES})
#message(${CMAKE_CURRENT_SOURCE_DIR})

add_library(Graphics STATIC
    ${GRAPHICS_SOURCES}
    #${GENERATED_SOURCES}
    ${GRAPHICS_SHADERS}
    #${GRAPHICS_UIS} 
)
 
target_link_libraries(Graphics
    ${GLEW_LIBRARIES} 
    Qt5::Core Qt5::Widgets Qt5::OpenGL Qt5::Gui
    ${VTK_LIBRARIES} 
    ${OPENGL_LIBRARIES} 
)

target_include_directories(Graphics PUBLIC ${PROJECT_SOURCE_DIR} ${GLM_DIR} )
