cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

# Set package properties
project(IO CXX)
#set(CMAKE_CXX_STANDARD 11)
#Set(CMAKE_CXX_FLAGS "/MT /EHsc")
 
# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)

###
# Perform build configuration of dependencies

# Set up QT5 and required components
find_package(Qt5 COMPONENTS Core Gui REQUIRED)

set (IO_SOURCES
        VolumeReader.cpp
        VolumeWriter.cpp
        PVMReader.cpp
    )

set (IO_HEADERS
        VolumeReader.h 
        VolumeWriter.h
        PVMReader.h
    )


# Let QT pre-process the files and add the generated files to the source list
#qt5_wrap_ui(GENERATED_SOURCES)

source_group("Generated Sources - Do Not Edit" FILES ${GENERATED_SOURCES})

# Make sure the compiler can find the pre-processed files from qt4_wrap_ui
include_directories("${CMAKE_BINARY_DIR}")
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/..")
#include_directories("C:/gitlab/vtk-8.2.0/Imaging/General")

#VTK
find_package(VTK REQUIRED)
if(VTK_VERSION_MAJOR EQUAL 8 AND VTK_VERSION_MINOR LESS 9)
    include(${VTK_USE_FILE})
endif()

#ITK
find_package(ITK REQUIRED) 
include(${ITK_USE_FILE})

#LIBICS
set(LIBICS_CPP_DIR "C:/Program\ Files/libics/cmake")
include_directories("${LIBICS_CPP_DIR}/../include")

# Build the app!
add_library(IO STATIC
     # source files that are actually built directly
     ${IO_SOURCES}
     #${GENERATED_SOURCES}
     # items included so they show up in your IDE
     ${IO_HEADERS}
    )

target_link_libraries(IO Qt5::Core Qt5::Gui ${ITK_LIBRARIES} ${VTK_LIBRARIES}
    debug ${LIBICS_CPP_DIR}/../lib/Debug/libics.lib
    debug ${LIBICS_CPP_DIR}/../lib/Debug/libics_cpp.lib
    optimized ${LIBICS_CPP_DIR}/../lib/Release/libics.lib
    optimized ${LIBICS_CPP_DIR}/../lib/Release/libics_cpp.lib
    )

target_compile_features(IO PRIVATE cxx_range_for)
