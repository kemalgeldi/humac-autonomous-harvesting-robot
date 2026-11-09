cmake_minimum_required(VERSION 3.0.2)
project(visual_servo)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  rospy
  sensor_msgs
  std_msgs
  tf2_geometry_msgs
  tf2_ros
  vision_msgs
  visualization_msgs
)

catkin_package(CATKIN_DEPENDS
  cv_bridge
  geometry_msgs
  rospy
  sensor_msgs
  std_msgs
  tf2_geometry_msgs
  tf2_ros
  vision_msgs
  visualization_msgs
)

catkin_install_python(PROGRAMS
  scripts/unified_mapper.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
