"""Phần 3: TurtleBot 4 behaviors driven by the recognized hand gestures (/detected_gesture).

State machine (published on /robot_behavior/state, shown in the webcam window):
    IDLE --Pointing_Up / Victory / Wave--> SELECTED --Thumb_Up--> RUNNING --done--> IDLE
    RUNNING --Open_Palm / lost signal--> PAUSED --Thumb_Up--> RUNNING (continues where it stopped)
    any state --Closed_Fist--> IDLE (behavior cancelled)

    Wave         greet: turn left-right-left-right, back to the start heading
    Pointing_Up  select forward_back: 4 x (forward, back)
    Victory      select circle: one circle
    Thumb_Up     confirm: run (or continue) the selected behavior
    Open_Palm    stop at once
    Closed_Fist  cancel, back to IDLE

A behavior is a list of segments sent on /cmd_vel at 20 Hz: (linear m/s, angular rad/s) for a time,
or for greet a turn to a heading relative to the start heading, closed-loop on /odom.
Against noisy recognition, a command fires only when the gesture changes (holding a gesture does
not repeat it), and select/confirm commands are ignored within `cooldown` s of the previous one.
The robot stops when no /detected_gesture message arrives for `signal_timeout` s, and when the
LiDAR sees an obstacle closer than `obstacle_distance` m in its direction of motion.

    ros2 run turtlebot4_bai06 robot_behavior_node
"""
import math
from collections import namedtuple

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

SELECT = {"Pointing_Up": "forward_back", "Victory": "circle", "Wave": "greet"}

# v, w: velocity command; time: duration (estimate for a turn); yaw: turn to this heading (rad,
# relative to the heading at start) at most w rad/s instead of driving for `time`.
Segment = namedtuple("Segment", "v w time yaw", defaults=(None,))


def build_behaviors(speed, distance, pause, repeats, radius, turns, turn_speed, turn_angle):
    move = distance / speed
    forward_back = [Segment(speed, 0.0, move), Segment(0.0, 0.0, pause),
                    Segment(-speed, 0.0, move), Segment(0.0, 0.0, pause)] * repeats
    w = speed / radius
    circle = [Segment(speed, w, turns * 2 * math.pi / w)]
    a = turn_angle / turn_speed
    greet = [Segment(0.0, turn_speed, a, turn_angle), Segment(0.0, turn_speed, 2 * a, -turn_angle),
             Segment(0.0, turn_speed, 2 * a, turn_angle), Segment(0.0, turn_speed, a, 0.0)]
    return {"forward_back": forward_back, "circle": circle, "greet": greet}


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class RobotBehaviorNode(Node):
    def __init__(self):
        super().__init__("robot_behavior_node")
        self.declare_parameter("speed", 0.2)               # m/s (forward_back, circle)
        self.declare_parameter("distance", 0.4)            # m, each forward / back move
        self.declare_parameter("pause", 0.5)               # s between moves
        self.declare_parameter("repeats", 4)
        self.declare_parameter("radius", 0.5)              # m, circle
        self.declare_parameter("turns", 1.0)               # circles
        self.declare_parameter("turn_speed", 1.0)          # rad/s, greet
        self.declare_parameter("turn_angle", 0.5)          # rad, greet: each side
        self.declare_parameter("wave_needs_confirm", True)  # Wave selects greet like the others
        self.declare_parameter("cooldown", 1.0)            # s between select/confirm commands
        self.declare_parameter("signal_timeout", 1.0)      # s without /detected_gesture -> stop
        self.declare_parameter("obstacle_distance", 0.3)   # m, 0 = no LiDAR check
        self.declare_parameter("scan_yaw", math.pi / 2)    # rplidar_link yaw in base_link (TurtleBot 4)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.behaviors = build_behaviors(p("speed"), p("distance"), p("pause"), int(p("repeats")), p("radius"),
                                         p("turns"), p("turn_speed"), p("turn_angle"))
        self.wave_confirm = bool(p("wave_needs_confirm"))
        self.cooldown = float(p("cooldown"))
        self.timeout = float(p("signal_timeout"))
        self.obstacle = float(p("obstacle_distance"))
        self.scan_yaw = float(p("scan_yaw"))

        self.state, self.behavior, self.reason = "IDLE", None, ""
        self.segment, self.seg_time = 0, 0.0          # progress in the behavior
        self.yaw, self.start_yaw = None, 0.0
        self.gesture, self.last_msg, self.last_cmd = "None", None, -1e9
        self.front = self.rear = math.inf
        self.zero_left = 0

        self.cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.state_pub = self.create_publisher(String, "robot_behavior/state", 10)
        self.create_subscription(String, "detected_gesture", self._on_gesture, 10)
        self.create_subscription(Odometry, "odom", self._on_odom, 10)
        if self.obstacle > 0:
            self.create_subscription(LaserScan, "scan", self._on_scan, qos_profile_sensor_data)
        self.dt = 0.05
        self.create_timer(self.dt, self._tick)
        self.create_timer(0.5, self._publish_state)

        # Create 3 refuses to drive backwards more than 0.15 m unless the safety override allows it.
        self.param_client = self.create_client(SetParameters, "motion_control/set_parameters")
        self.override_timer = self.create_timer(2.0, self._allow_backup)
        self.get_logger().info("Waiting for gestures on /detected_gesture")

    # ---------------------------------------------------------------- gestures
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_gesture(self, msg):
        now = self._now()
        self.last_msg = now
        g = msg.data
        if g == self.gesture:      # held gesture: already handled
            return
        self.gesture = g
        if g == "Open_Palm":
            self._stop("Open_Palm")
        elif g == "Closed_Fist":
            self._cancel()
        elif g in SELECT or g == "Thumb_Up":
            if now - self.last_cmd < self.cooldown:
                self.get_logger().info(f"{g} ignored (cooldown)")
                return
            self.last_cmd = now
            if g == "Thumb_Up":
                self._confirm()
            else:
                self._select(SELECT[g], start=(g == "Wave" and not self.wave_confirm))

    def _select(self, name, start=False):
        if self.state == "RUNNING":
            self.get_logger().info(f"Busy with {self.behavior}: Open_Palm or Closed_Fist first")
            return
        self.behavior, self.segment, self.seg_time = name, 0, 0.0
        if start:
            self.start_yaw = self.yaw or 0.0
            self._set_state("RUNNING", "Wave")
        else:
            self._set_state("SELECTED", "Thumb_Up to start")

    def _confirm(self):
        if self.state in ("SELECTED", "PAUSED"):
            if self.state == "SELECTED":
                self.start_yaw = self.yaw or 0.0
            self._set_state("RUNNING", "confirmed")
        elif self.state == "IDLE":
            self.get_logger().info("Thumb_Up: no behavior selected")

    def _stop(self, reason):
        if self.state == "RUNNING":
            self._set_state("PAUSED", reason)
        self._send_zero()

    def _cancel(self):
        if self.state != "IDLE":
            self.behavior = None
            self._set_state("IDLE", "Closed_Fist: cancelled")
        self._send_zero()

    def _set_state(self, state, reason):
        self.state, self.reason = state, reason
        self.get_logger().info(f"{state} {self.behavior or ''} ({reason})")
        self._publish_state()

    def _publish_state(self):
        text = self.state
        if self.behavior:
            segments = self.behaviors[self.behavior]
            done = sum(s.time for s in segments[:self.segment])
            if self.segment < len(segments):
                done += min(self.seg_time, segments[self.segment].time)
            text += f" {self.behavior} {100 * done / sum(s.time for s in segments):.0f}%"
        if self.state in ("SELECTED", "PAUSED"):
            text += " - Thumb_Up to start" if self.state == "SELECTED" else f" ({self.reason}) - Thumb_Up to continue"
        self.state_pub.publish(String(data=text))

    def _on_odom(self, odom):
        q = odom.pose.pose.orientation
        self.yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    # ---------------------------------------------------------------- safety
    def _on_scan(self, scan):
        front = rear = math.inf
        for i, r in enumerate(scan.ranges):
            if not scan.range_min < r < scan.range_max:
                continue
            a = scan.angle_min + i * scan.angle_increment + self.scan_yaw   # angle in base_link
            a = wrap(a)
            if abs(a) < math.radians(30):
                front = min(front, r)
            elif abs(a) > math.radians(150):
                rear = min(rear, r)
        self.front, self.rear = front, rear

    def _allow_backup(self):
        if not self.param_client.service_is_ready():
            return
        value = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value="backup_only")
        self.param_client.call_async(SetParameters.Request(
            parameters=[Parameter(name="safety_override", value=value)]))
        self.get_logger().info("motion_control: safety_override = backup_only")
        self.override_timer.cancel()

    # ---------------------------------------------------------------- motion
    def _tick(self):
        now = self._now()
        if self.state == "RUNNING" and (self.last_msg is None or now - self.last_msg > self.timeout):
            self._stop("lost signal")
        if self.state != "RUNNING":
            if self.zero_left > 0:     # a few zero commands so the robot surely stops
                self.zero_left -= 1
                self.cmd_pub.publish(Twist())
            return

        segments = self.behaviors[self.behavior]
        while self.segment < len(segments) and self._segment_done(segments[self.segment]):
            self.segment, self.seg_time = self.segment + 1, 0.0
        if self.segment >= len(segments):
            name = self.behavior
            self.behavior = None
            self._set_state("IDLE", f"{name} done")
            self._send_zero()
            return

        v, w, _, yaw = segments[self.segment]
        if yaw is not None:   # proportional turn, slower near the target heading
            err = wrap(self.start_yaw + yaw - self.yaw) if self.yaw is not None else 0.0
            w = math.copysign(min(w, max(0.15, 2.5 * abs(err))), err) if err else 0.0
        if (v > 0 and self.front < self.obstacle) or (v < 0 and self.rear < self.obstacle):
            self._stop("obstacle")
            return
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = v, w
        self.cmd_pub.publish(cmd)
        self.seg_time += self.dt

    def _segment_done(self, seg):
        if seg.yaw is None:
            return self.seg_time >= seg.time
        if self.seg_time > 3 * seg.time:   # no odometry or stuck: give up this turn
            return True
        return self.yaw is not None and abs(wrap(self.start_yaw + seg.yaw - self.yaw)) < 0.03

    def _send_zero(self):
        self.cmd_pub.publish(Twist())
        self.zero_left = 5


def main():
    rclpy.init()
    node = RobotBehaviorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
