cmake_minimum_required(VERSION 3.16)
project(jennifer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)

include_directories(
    ${GLOG_INCLUDE_DIR}
    ${ARMADILLO_INCLUDE_DIRS}
    include)

file(GLOB_RECURSE SOURCE_CODE source/*.cpp)
add_library(jennifer SHARED ${SOURCE_CODE})
target_link_libraries(jennifer ${GLOG_LIBRARIES} ${ARMADILLO_LIBRARIES})

add_subdirectory(test)
add_subdirectory(demo)
