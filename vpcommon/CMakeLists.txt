cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(vpcommon CXX)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)

# Set up QT5 and required components
find_package(Qt5 COMPONENTS Core Widgets OpenGL Gui REQUIRED)

set (VPCOMMON_SOURCES
        vpCanvas.cpp
        vpCanvas.h
        vpDbsRect.cpp
        vpDbsRect.h
        vpDoubleSlider.cpp
        vpDoubleSlider.h
        vpDragableEnd.cpp
        vpDragableEnd.h
        vpStructures.cpp
        vpStructures.h
        vpTextureContext.cpp
        vpTextureContext.h
        vpTFNode.cpp
        vpTFNode.h
        vpTFQuad.cpp
        vpTFQuad.h
        vpTFScene.cpp
        vpTFScene.h
        vpTypes.h
        vpTransferFunction.cpp
        vpTransferFunction.h
        vpTransferFunctionEditor.cpp
        vpTransferFunctionEditor.h
        #vpVolumeStats.h
    )

set (VPCOMMON_UIS )

#set (VPCOMMON_SHADERS
#    cube.vert.glsl
#    cube.frag.glsl
#    slice.vert.glsl
#    slice.frag.glsl
#    raycasting.vert.glsl
#    raycasting.frag.glsl
#)


# Let QT pre-process the files and add the generated files to the source list
qt5_wrap_ui(VPCOMMON_GENSOURCES ${VPCOMMON_UIS})

source_group("Generated Sources - Do Not Edit" FILES ${VPCOMMON_GENSOURCES})

# Make sure the compiler can find the pre-processed files from qt4_wrap_ui
set( ThirdParty_DIR "${PROJECT_SOURCE_DIR}/../../ThirdParty")
include_directories("${CMAKE_BINARY_DIR}") 
include_directories("${PROJECT_SOURCE_DIR}")
#include_directories("${PROJECT_SOURCE_DIR}/../cline")
#include_directories("${PROJECT_SOURCE_DIR}/../vpapp")
include_directories("${PROJECT_SOURCE_DIR}/..")
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${ThirdParty_DIR}/glm-0.9.8.4")
include_directories("${ThirdParty_DIR}/colorbrewercpp")

find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIR})
find_package(OpenGL REQUIRED) 
include_directories( ${OPENGL_INCLUDE_DIRS} )

find_package(VTK REQUIRED)
if(VTK_VERSION_MAJOR EQUAL 8 AND VTK_VERSION_MINOR LESS 9)
    include(${VTK_USE_FILE})
endif()

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})


# Build the app!
add_library(vpcommon STATIC 
     ${VPCOMMON_SOURCES}
     ${VPCOMMON_GENSOURCES}
     # items included so they show up in your IDE
     ${VPCOMMON_HEADERS}
     ${VPCOMMON_UIS}
     #${VPCOMMON_SHADERS}
)

# Link the app!
target_link_libraries(vpcommon 
    Qt5::Core Qt5::Widgets Qt5::OpenGL Qt5::Gui 
    ${GLEW_LIBRARIES} 
    ${OPENGL_LIBRARIES}
    ${VTK_LIBRARIES}
    ${ITK_LIBRARIES}
    )

target_include_directories(vpcommon PUBLIC ${PROJECT_SOURCE_DIR})
