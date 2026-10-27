# turtlebot4_bai06 — Bài 06: controlling TurtleBot 4 with hand gestures

TurtleBot 4 Lite in Ignition Gazebo (ROS 2 Humble) driven by hand gestures from the laptop webcam:
OpenCV reads the webcam, the MediaPipe Gesture Recognizer classifies the hand, and a state machine
turns the gestures into `/cmd_vel` behaviors. Simulation from
[bài 04](https://github.com/Khanhkhan1/turtlebot4_bai04); no map or Nav2.

**Video demo:** https://drive.google.com/file/d/1ZBZYG4bKOdI0HT6nV3zjbc9GY0d5UPN2/view?usp=sharing

```
webcam --OpenCV--> gesture_recognition_node (MediaPipe + wave detection + debounce)
       --/detected_gesture (std_msgs/String, 10 Hz)--> robot_behavior_node (state machine)
       --/cmd_vel--> TurtleBot 4 (Gazebo)        /odom --> robot_behavior_node (greet turns)
robot_behavior_node --/robot_behavior/state--> webcam window
Gazebo overview camera --/overview_camera/image--> RViz2 ("Gazebo view")
```

## Gestures

| Gesture | Robot |
|---|---|
| Wave (open hand swung left-right) | selects *greet*: turn left-right-left-right |
| Pointing_Up | selects *forward_back*: 4 × (0.4 m forward, 0.4 m back) |
| Victory | selects *circle*: one circle, radius 0.5 m |
| Thumb_Up | confirms: runs the selected behavior (or continues a stopped one) |
| Open_Palm | stops at once (Thumb_Up continues) |
| Closed_Fist | cancels, back to IDLE |

The robot moves only after Thumb_Up. A gesture counts once it is held for 0.5 s (Open_Palm 0.2 s;
Closed_Fist 0.6 s, as the hand passes through a fist between two gestures) and only when it
changes; select/confirm commands have a 1 s cooldown.
The robot stops when `/detected_gesture` is silent for 1 s (camera or node lost).
`wave_needs_confirm:=false` makes Wave start the greeting directly.

## Contents

- `turtlebot4_bai06/gesture_recognition_node.py` — webcam → MediaPipe (`gesture_recognizer.task`,
  downloaded to `~/.cache/turtlebot4_bai06`) → landmarks, gesture and confidence in the OpenCV window,
  stable gesture on `/detected_gesture`. Wave is not a MediaPipe class: it is detected from the
  fingertip motion of an open hand (≥ 3 direction changes within 1.5 s).
- `turtlebot4_bai06/robot_behavior_node.py` — IDLE / SELECTED / RUNNING / PAUSED state machine,
  behaviors as `/cmd_vel` segments at 20 Hz, watchdog.
- `turtlebot4_bai06/webcam_view.py` — Phần 1: show the webcam with OpenCV.
- `launch/bai06.launch.py` — bài 04 simulation in `worlds/open_field.sdf` (8 × 8 m), overview camera,
  RViz2 (`rviz/bai06.rviz`), both nodes.

## Build

```bash
cd ~/tb4_ws/src && git clone https://github.com/Khanhkhan1/turtlebot4_bai06.git
vcs import . < turtlebot4_bai06/dependencies.repos   # turtlebot4_bai04, turtlebot4, simulator, create3_sim
pip install mediapipe
cd ~/tb4_ws && colcon build && source install/setup.bash
```

## Run

```bash
export ROS_LOCALHOST_ONLY=1 IGN_IP=127.0.0.1          # in every terminal

ros2 run turtlebot4_bai06 webcam_view                  # Phần 1: webcam check (q to quit)
ros2 launch turtlebot4_bai06 bai06.launch.py           # Gazebo, RViz2, both nodes (camera:=1 for another webcam)

# Phần 1: /cmd_vel check, or the behaviors without a webcam
ros2 launch turtlebot4_bai06 bai06.launch.py gesture:=false
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.1}}"
ros2 topic pub -r 10 /detected_gesture std_msgs/msg/String "{data: Victory}"   # then Thumb_Up, ...
```
