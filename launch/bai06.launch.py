#!/usr/bin/env python3
"""Bài 06 bring-up: TurtleBot 4 in an open area (open_field.sdf), a fixed overview camera that
shows the Gazebo scene, RViz2, robot_behavior_node and gesture_recognition_node (webcam).

The simulation itself is turtlebot4_bai04/sim.launch.py (Gazebo headless on macOS, bridges,
Create 3 nodes); no SLAM, map or Nav2 here.

    ros2 launch turtlebot4_bai06 bai06.launch.py                   # everything
    ros2 launch turtlebot4_bai06 bai06.launch.py gesture:=false    # robot only (test with ros2 topic pub)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument, IncludeLaunchDescription,
                            SetLaunchConfiguration, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# Overview camera pose (x y z roll pitch yaw): behind-left of the start point, looking at it.
CAMERA_POSE = ["-1.3", "-0.75", "1.15", "0", "0.55", "0.71"]


def generate_launch_description():
    pkg = get_package_share_directory("turtlebot4_bai06")
    pkg_bai04 = get_package_share_directory("turtlebot4_bai04")

    args = [
        DeclareLaunchArgument("rviz", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("gesture", default_value="true", choices=["true", "false"],
                              description="Start gesture_recognition_node (webcam)"),
        DeclareLaunchArgument("camera", default_value="0", description="Webcam index"),
        DeclareLaunchArgument("wave_needs_confirm", default_value="true", choices=["true", "false"],
                              description="false: Wave starts the greeting without Thumb_Up"),
    ]

    # Gazebo looks for open_field.sdf here.
    world_path = [AppendEnvironmentVariable(name, os.path.join(pkg, "worlds"))
                  for name in ("IGN_GAZEBO_RESOURCE_PATH", "GZ_SIM_RESOURCE_PATH")]

    # The include sets rviz:=false for the whole launch (no scoping in Humble): keep ours first.
    keep_rviz = SetLaunchConfiguration("bai06_rviz", LaunchConfiguration("rviz"))
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_bai04, "launch", "sim.launch.py")),
        launch_arguments={"world": "open_field", "rviz": "false"}.items(),
    )

    # Gazebo's Sensors system comes with the robot (Create 3 description): add the camera after it.
    overview_camera = TimerAction(period=12.0, actions=[
        Node(
            package="ros_gz_sim", executable="create", output="log",
            arguments=["-file", os.path.join(pkg, "worlds", "overview_camera.sdf"), "-name", "overview_camera",
                       "-x", CAMERA_POSE[0], "-y", CAMERA_POSE[1], "-z", CAMERA_POSE[2],
                       "-R", CAMERA_POSE[3], "-P", CAMERA_POSE[4], "-Y", CAMERA_POSE[5]],
        ),
        Node(
            package="ros_gz_bridge", executable="parameter_bridge", name="overview_camera_bridge", output="log",
            arguments=["/overview_camera/image@sensor_msgs/msg/Image[ignition.msgs.Image"],
        ),
    ])

    # On macOS the RViz2 OpenGL renderer can segfault while the simulation starts: respawn it.
    rviz = Node(
        package="rviz2", executable="rviz2", name="rviz2", output="log",
        arguments=["-d", os.path.join(pkg, "rviz", "bai06.rviz")],
        parameters=[{"use_sim_time": True}],
        respawn=True, respawn_delay=2.0,
        condition=IfCondition(LaunchConfiguration("bai06_rviz")),
    )

    behavior = Node(
        package="turtlebot4_bai06", executable="robot_behavior_node", name="robot_behavior_node", output="screen",
        parameters=[{"use_sim_time": True,
                     "wave_needs_confirm": ParameterValue(LaunchConfiguration("wave_needs_confirm"), value_type=bool)}],
    )
    gesture = Node(
        package="turtlebot4_bai06", executable="gesture_recognition_node", name="gesture_recognition_node",
        output="screen",
        parameters=[{"camera": ParameterValue(LaunchConfiguration("camera"), value_type=int)}],
        condition=IfCondition(LaunchConfiguration("gesture")),
    )

    return LaunchDescription(args + world_path + [keep_rviz, sim, overview_camera, rviz, behavior,
                                                  TimerAction(period=15.0, actions=[gesture])])
