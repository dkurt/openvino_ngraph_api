cmake_minimum_required(VERSION 3.4.3)

project(ngraph_demo CXX)

find_package(OpenCV REQUIRED COMPONENTS dnn highgui)
find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  include
)

file(GLOB SOURCES src/*.cpp)
file(GLOB HEADERS include/*.hpp)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCES} ${HEADERS})
target_compile_features(${CMAKE_PROJECT_NAME} PRIVATE cxx_range_for)

message(${InferenceEngine_LIBRARIES})

target_link_libraries(${CMAKE_PROJECT_NAME}
  ${OpenCV_LIBS}
  ${InferenceEngine_LIBRARIES}
  ${NGRAPH_LIBRARIES}
)
