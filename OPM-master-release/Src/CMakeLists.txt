project(OPM_linux)
cmake_minimum_required(VERSION 2.8)

find_package(OpenMP REQUIRED)

include_directories(${Boost_INCLUDE_DIR}
		           /usr/include/python3.4)		    

#include_directories(${PYTHON_INCLUDE_DIRS})

#find_package(PythonLibs REQUIRED)
#include_directories("/usr/include/python2.7")
include_directories("Synthesize")
#file(GLOB SOURCES "Synthesize/*.cpp")
#add_library(Synthesize SHARED ${SOURCES})   #STATIC  or SHARED
# compile LibLinear
include_directories("LibLinear")
file(GLOB SOURCES "LibLinear/*.cpp" "LibLinear/blas/*.c")
add_library(LibLinear STATIC ${SOURCES})
set(OpenCV_DIR "/usr/local/opencv3.2.0/share/OpenCV") 
#OPENCV
include_directories(/usr/local/opencv3.2.0/include)
link_directories(/usr/local/opencv3.2.0/lib)
##if this does not work, then try to uncomment the things below.
#find_package( OpenCV REQUIRED )
find_package(OpenCV 3.2 REQUIRED)
if(OpenCV_FOUND)
 include_directories( ${OpenCV_INCLUDE_DIRS} )
endif( OpenCV_FOUND )
list( APPEND CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS} -fopenmp -ftree-vectorize ")

# compile OPM
file(GLOB SOURCES "*.cpp")
add_library(OPM STATIC ${SOURCES})
#

add_executable(${PROJECT_NAME} main.cpp)

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")
#target_link_libraries(${PROJECT_NAME} "/usr/lib/python3.4/config-3.4m-x86_64-linux-gnu/libpython3.4.so")
target_link_libraries(${PROJECT_NAME}  opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs ${EXTERNAL_LIBS} OPM LibLinear /home/yourpath/Documents/OPM-master-release/Src/libMatplot.a /home/yourpath/Documents/OPM-master-release/Src/libSynthesize.so)
target_link_libraries(${PROJECT_NAME} /usr/lib/x86_64-linux-gnu/libstdc++.so.6  /usr/lib/x86_64-linux-gnu/libpython3.4m.so) 
