project(test_jennifer)
find_package(GTest REQUIRED)

file(GLOB TEST_SOURCES "*.cpp")
add_executable(${PROJECT_NAME} ${TEST_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC ${GTest_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} GTest::gtest jennifer)