cmake_minimum_required(VERSION 2.6)
project(camera_imu_calibration)

add_definitions(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  nav_msgs
  sensor_msgs
  rospy
  roscpp
  image_transport
  cv_bridge
)
find_package(OpenCV REQUIRED)
find_package(GTSAM)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES
  CATKIN_DEPENDS roscpp rospy nav_msgs std_msgs sensor_msgs
)

###########
## Build ##
###########
include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${GTSAM_INCLUDE_DIR}
)


add_executable(camera_imu_calibration
  src/camera_imu_calibration.cpp
  include/camera_imu_calibration.h
)
target_link_libraries(camera_imu_calibration ${OpenCV_LIBS} ${catkin_LIBRARIES} gtsam)

add_executable(create_board
    src/create_board.cpp
)
target_link_libraries(create_board ${catkin_LIBRARIES} ${OpenCV_LIBS})
