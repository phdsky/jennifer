project(jennifer_yolo)

find_package(OpenCV REQUIRED)

file(GLOB TEST_SOURCES "*.cpp")
add_executable(${PROJECT_NAME} ${TEST_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} jennifer)