import rclpy
import cv2
import numpy as np
from auto_aim_interfaces.srv import TrackingMode
from auto_aim_interfaces.msg import Leafs
from .utils.angleProcessor import bigPredictor, smallPredictor, angleObserver, trans, clock, mode
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from auto_aim_interfaces.msg import Tracker2D
from auto_aim_interfaces.msg import EnTarget
import tf2_ros
from geometry_msgs.msg import PoseStamped
from tf2_geometry_msgs import do_transform_pose
from rclpy.duration import Duration
from builtin_interfaces.msg import Time


class energy_tracker(Node):
    def __init__(self, options):
        super().__init__(options)
        self.get_logger().info("<节点初始化> 能量机关预测器")
        self.predict_mode_service = self.create_service(
            TrackingMode, "EnTracker/resetmode", self.predict_mode_service_callback)  # 预测模式服务端
        self.get_logger().info("<节点初始化> 能量机关预测器/预测模式服务端")

        self.Leafs_Sub = self.create_subscription(
            Leafs, "detector/leafs", self.LeafsCallback, rclpy.qos.qos_profile_sensor_data)
        self.get_logger().info("<节点初始化> 能量机关预测器/识别点订阅者")

        self.Target_pub2D = self.create_publisher(
            Tracker2D, "tracker/LeafTarget2D", rclpy.qos.qos_profile_sensor_data)
        self.get_logger().info("<节点初始化> 能量机关预测器/2D预测点发布者")

        self.Target_pub = self.create_publisher(
            EnTarget, "tracker/LeafTarget", rclpy.qos.qos_profile_sensor_data)
        self.get_logger().info("<节点初始化> 能量机关预测器/目标信息发布者")

        # init info
        self.is_start = True
        self.moveMode = mode.big
        self.v = 710  # m/s
        # self.freq = 50
        self.color = self.declare_parameter("detect_color", "BLUE").value
        self.color = self.get_parameter(
            "detect_color").get_parameter_value().string_value
        self.observer = angleObserver(
            clockMode=clock.anticlockwise) if self.color == "BLUE" else angleObserver(clockMode=clock.clockwise)
        # self.observer=angleObserver(clockMode=clock.anticlockwise)
        # time info
        self.time = Time()
        self.lasttime = Time()
        self.frame_pass = 2

        # tf2 settings
        self.tf2_buffer_ = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer_, self)
        self.target_frame = self.declare_parameter(
            "target_frame", "odom").value
        self.debug_ = self.declare_parameter(
            "debug", value=False).get_parameter_value().bool_value
        # tracking info
        self.lastPosition = None

    def predict_mode_service_callback(self, mode_request, mode_response):
        if mode_request.mode == 1:
            self.moveMode = mode.small
        elif mode_request.mode == 2:
            self.moveMode = mode.big
        elif mode_request.mode == 0:
            self.moveMode = mode.person
        self.is_start = True
        mode_response.success = True
        self.time = Time()
        self.lasttime = Time()
        self.frame_pass = 2
        return mode_response

    def GetpreLen(self, leaf, pre_leaf_center2D, deltaAngle):
        leaf_center2D = (leaf.leaf_center.z, leaf.leaf_center.y)
        angle = np.deg2rad(deltaAngle)
        b = 2*np.cos(angle)*leaf.distance_to_image_center
        a = 1
        c = -(leaf.distance_to_image_center*leaf.distance_to_image_center+cv2.norm(
            leaf_center2D, pre_leaf_center2D)*cv2.norm(leaf_center2D, pre_leaf_center2D))
        len_1 = (-b+np.sqrt(b*b-4*a*c))/(2*a)
        len_2 = (-b-np.sqrt(b*b-4*a*c))/(2*a)
        if len_1 < 0 and len_2 < 0:
            return leaf.distance_to_image_center
        elif len_1 < 0:
            return len_2
        elif len_2 < 0:
            return len_1

    def Gravity_compensation(self, bottom_len, v, angle_0) -> float():
        g = 9.788  # m/s
        t = bottom_len/(v*np.cos(np.deg2rad(angle_0)))
        h = g*t**2/2
        return float(h)

    def LeafsCallback(self, leafs_msg):
        # find the max prob's leaf
        if (len(leafs_msg.leafs) > 0):
            leaf_ = leafs_msg.leafs[0]
            for leaf in leafs_msg.leafs:
                if leaf.prob > leaf_.prob:
                    leaf_ = leaf
            # define the freq and dt
            if self.frame_pass != 0 and self.is_start is True:
                self.time = leafs_msg.header.stamp
                self.dt = self.time.nanosec-self.lasttime.nanosec
                self.lasttime = self.time
                self.frame_pass -= 1
                return
            # init the predict_tracker
            if self.is_start is True:
                self.dt = float(self.dt/1e9)
                self.dt_error = 0.05
                self.freq = int(1.0/self.dt)*10
                self.freq = 80  # fixed freq(fps)
                self.get_logger().info("Fixed_dt={},Fix_Fps={}".format(self.dt, self.freq))

                if self.moveMode == mode.small:
                    self.predictor = smallPredictor(
                        freq=self.freq, deltaT=self.dt+self.dt_error)
                elif self.moveMode == mode.big:
                    self.predictor = bigPredictor(
                        freq=self.freq, deltaT=self.dt+self.dt_error)
                interval = int(self.freq * self.dt+self.dt_error)
                self.is_start = False

                A_p = np.array([leaf_.leaf_center.z, leaf_.leaf_center.y])
                R_p = np.array([leaf_.r_center.z, leaf_.r_center.y])
                x, y = A_p - R_p  # 分别算出二维r中心与扇叶中心的x,y距离
                self.radius = np.sqrt(x**2+y**2)

            A_p = np.array([leaf_.leaf_center.z, leaf_.leaf_center.y])
            R_p = np.array([leaf_.r_center.z, leaf_.r_center.y])
            x, y = A_p - R_p  # 分别算出二维r中心与扇叶中心的x,y距离
            angle = self.observer.update(x, y, self.radius)  # 角度更新
            flag, deltaAngle = self.predictor.update(angle)
            if flag:
                angle = trans(x, y) + deltaAngle
                x = np.cos(angle) * self.radius  # 提前x 秒后的扇叶中心的x
                y = np.sin(angle) * self.radius  # 提前x 秒后的扇叶中心的y
                x, y = np.array([x, y]) + R_p  # 得到最终的预测扇叶中心

                Target2d = Tracker2D()
                Target2d.x = float(x)
                Target2d.y = float(y)
                self.Target_pub2D.publish(Target2d)

                if self.debug_:
                    self.get_logger().info("predict_x={},predict_y={}".format(x, y))

                angle = np.deg2rad(deltaAngle)
                if leaf_.type == "INVALID" and self.lastPosition != None:
                    leaf_pose.position = self.lastPosition
                x = leaf_.pose.position.x
                y = leaf_.pose.position.y
                z = leaf_.pose.position.z
                leaf_.pose.position.x = x*np.cos(angle)-z*np.sin(angle)
                leaf_.pose.position.z = x*np.sin(angle)+z*np.cos(angle)
                self.lastPosition = leaf_.pose.position
                # tf2 trasform
                ps = PoseStamped()
                ps.header = leafs_msg.header
                ps.pose = leaf_.pose
                try:
                    transf = self.tf2_buffer_.lookup_transform(
                        "odom", "gimbal_link", rclpy.time.Time())
                    leaf_.pose = do_transform_pose(leaf_.pose, transf)
                except:
                    self.get_logger().error("Error while transforming {}".format(
                        tf2_ros.ExtrapolationException()))
                    return

                # yaw角和pitch角度的解算
                Target = EnTarget()
                px = leaf_.pose.position.x
                py = leaf_.pose.position.y
                pz = leaf_.pose.position.z
                self.get_logger().info("px={},py={},pz={}".format(px, py, pz))
                bottom_len = np.sqrt(py**2+px**2)
                Target.pitch = np.rad2deg(
                    np.arctan2(pz, bottom_len))  # pitch angle
                Target.pitch = Target.pitch = np.rad2deg(
                    np.arctan2(pz+self.Gravity_compensation(bottom_len=bottom_len, v=self.v, angle_0=Target.pitch), bottom_len))

                Target.yaw = np.rad2deg(np.arctan2(px, py))  # yaw angle
                Target.position.x, Target.position.y, Target.position.z = px, py, pz
                Target.header.stamp = leafs_msg.header.stamp
                Target.header.frame_id = self.target_frame
                if self.debug_:
                    self.get_logger().info("pitch={},yaw={}".format(Target.pitch, Target.yaw))

                self.Target_pub.publish(Target)


def main(args=None):
    rclpy.init(args=args)
    node = energy_tracker("energy_tracker_node")
    rclpy.spin(node)
    rclpy.shutdown()
