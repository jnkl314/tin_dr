CMAKE_MINIMUM_REQUIRED(VERSION 3.2)
project(TIN_DR_modules)
set(TARGET_NAME ${PROJECT_NAME})

######################################
########### CMake Options ############
######################################
option(TIN_DR_WITH_VERBOSE "Enable verbose mode" ON)

set(OPENCV_VERSION "" CACHE STRING "OpenCV Version to specify")

######################################
########### Create target ############
######################################
# List source files
file(GLOB_RECURSE LOCAL_SOURCES src/*.cpp)
# List header files
file(GLOB_RECURSE LOCAL_HEADERS include/*.hpp private_include/*.hpp)
# Create Lib
add_library(${PROJECT_NAME} ${LOCAL_SOURCES} ${LOCAL_HEADERS})
target_include_directories(${TARGET_NAME}
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/private_include)

######################################
######### Add OpenCV Library #########
######################################
find_package(OpenCV ${OPENCV_VERSION} REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS})

######################################
########### Build Options ############
######################################
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fpic -Wall")

######################################
########## Add Definitions ###########
######################################
if(TIN_DR_WITH_VERBOSE)
    target_compile_definitions(${TARGET_NAME} PUBLIC TIN_DR_ENABLE_VERBOSE)
endif(TIN_DR_WITH_VERBOSE)
