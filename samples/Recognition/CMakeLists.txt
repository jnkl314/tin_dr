CMAKE_MINIMUM_REQUIRED(VERSION 3.2)
project(TextRecognition)

######################################
########### CMake Options ############
######################################
set(OPENCV_VERSION "" CACHE STRING "OpenCV Version to specify")

######################################
########### Create target ############
######################################
# Create Exe
add_executable(${PROJECT_NAME} main.cpp)

######################################
############ Add modules  ############
######################################
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../modules ${CMAKE_BINARY_DIR}/modules)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../modules/include/api)
target_link_libraries(${PROJECT_NAME} TIN_DR_modules)

######################################
######### Add OpenCV Library #########
######################################
find_package(OpenCV ${OPENCV_VERSION} REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

######################################
########### Build Options ############
######################################
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fpic -Wall -pthread")

