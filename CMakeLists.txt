cmake_minimum_required(VERSION 2.6)

add_definitions(-std=c++11)


find_package(OpenCV REQUIRED)

include_directories(include)

add_executable(camera_imu_calibration 
  src/camera_imu_calibration.cpp
  include/camera_imu_calibration.h
)
target_link_libraries(camera_imu_calibration ${OpenCV_LIBS})

add_executable(create_board
    src/create_board.cpp
)
target_link_libraries(create_board ${OpenCV_LIBS})
