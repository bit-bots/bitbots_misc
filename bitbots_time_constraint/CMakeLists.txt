cmake_minimum_required(VERSION 2.8.3)
project(bitbots_time_constraint)

find_package(catkin REQUIRED COMPONENTS
 rospy
 bitbots_docs
)

catkin_python_setup()

catkin_package()

enable_bitbots_docs()
