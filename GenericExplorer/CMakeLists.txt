cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

# Set package properties
project(GenericExplorer CXX)
set(CMAKE_CXX_STANDARD 11)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)

###
# Perform build configuration of dependencies
# Set up QT5 and required components
find_package(Qt5 COMPONENTS Core Gui OpenGL Widgets WebEngine WebEngineWidgets WebChannel REQUIRED)

set (GenericExplorer_Sources
        GenericExplorer_main.cpp
        Bookmark.cpp
        Bookmark.h
        BookmarkListModel.cpp
        BookmarkListModel.h
        BookmarkTool.cpp
        BookmarkTool.h
        DialogOpticalProperties.cpp
        DialogOpticalProperties.h
        VolumeVisualizer.cpp
        VolumeVisualizer.h
        VolumeStats.cpp
        VolumeStats.h
        ../../Bhattacharya/Slider.cpp
        ../../Bhattacharya/Slider.h
    )

set (EXTRA_FILES
	)

source_group("Generated Sources - Do Not Edit" FILES ${GENERATED_SOURCES})

# Make sure the compiler can find the pre-processed files from qt4_wrap_ui
set(ThirdParty_DIR "${PROJECT_SOURCE_DIR}/../ThirdParty")
include_directories("${CMAKE_BINARY_DIR}")
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/../fvis")
include_directories("${PROJECT_SOURCE_DIR}/../fvis/GraphSeg")
include_directories("${PROJECT_SOURCE_DIR}/../fvis/io")
include_directories("${PROJECT_SOURCE_DIR}/../fvis/ds")
include_directories(${ThirdParty_DIR})
include_directories("${PROJECT_SOURCE_DIR}/../vp/vpcommon")
include_directories("${ThirdParty_DIR}/glm-0.9.8.4")

add_subdirectory(${PROJECT_SOURCE_DIR}/../fvis/GraphSeg libGraphSeg)
add_subdirectory(${PROJECT_SOURCE_DIR}/../fvis//core libCore)
add_subdirectory(${PROJECT_SOURCE_DIR}/../fvis/ds libDS)
add_subdirectory(${PROJECT_SOURCE_DIR}/../fvis/io libIO)
add_subdirectory(${PROJECT_SOURCE_DIR}/../fvis/graphics libGraphics)
add_subdirectory(${PROJECT_SOURCE_DIR}/../vp/vpcommon libvpcommon)
add_subdirectory(${PROJECT_SOURCE_DIR}/../fvis/utils libUtils)
link_directories(C:/Anaconda3/Library/lib)
#link_directories(${PROJECT_SOURCE_DIR}/../../armadillo-7.950.1/examples/lib_win64)

find_package(VTK REQUIRED)
if(VTK_VERSION_MAJOR EQUAL 8 AND VTK_VERSION_MINOR LESS 9)
    include(${VTK_USE_FILE})
endif()
find_package(OpenGL REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS} )
find_package(glew REQUIRED)
include_directories( ${GLEW_INCLUDE_DIR} )

#LIBICS
set(LIBICS_CPP_DIR "C:/Program\ Files/libics/cmake")
include_directories("${LIBICS_CPP_DIR}/../include")

add_executable(GenericExplorer
     ${GENERATED_SOURCES}
     ${GenericExplorer_Sources}
     ${EXTRA_FILES} 
    )

target_link_libraries(GenericExplorer Core DS IO GraphSeg Graphics vpcommon Utils
	${GLEW_LIBRARIES} Qt5::Core Qt5::Widgets Qt5::OpenGL Qt5::Gui Qt5::WebEngine Qt5::WebEngineWidgets Qt5::WebChannel
    debug ${LIBICS_CPP_DIR}/../lib/Debug/libics.lib
    debug ${LIBICS_CPP_DIR}/../lib/Debug/libics_cpp.lib
    optimized ${LIBICS_CPP_DIR}/../lib/Release/libics.lib
    optimized ${LIBICS_CPP_DIR}/../lib/Release/libics_cpp.lib
    icuuc icudt zlib iphlpapi netapi32
    ${VTK_LIBRARIES} ${OPENGL_LIBRARIES} ) 

target_compile_features(GenericExplorer PRIVATE cxx_range_for)
