#include "energy_tracker/energy_tracker_node.hpp"

// STD
#include <memory>
#include <vector>

namespace rm_auto_aim
{
   EnergyTrackerNode::EnergyTrackerNode(const rclcpp::NodeOptions &options)
       : Node("energy_tarcker", options)
   {
      RCLCPP_INFO(this->get_logger(), "Starting EnergyTarckerNode!");
      // Subscriber with tf2 message_filter
      // tf2 relevant
      tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
      // Create the timer interface before call to waitForTransform,
      // to avoid a tf2_ros::CreateTimerInterfaceException exception
      auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
          this->get_node_base_interface(), this->get_node_timers_interface());
      tf2_buffer_->setCreateTimerInterface(timer_interface);
      tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);
      // subscriber and filter
      leafs_sub_.subscribe(this, "/detector/leafs", rmw_qos_profile_sensor_data);
      target_frame_ = this->declare_parameter("target_frame", "odom");
      tf2_filter_ = std::make_shared<tf2_filter>(
          leafs_sub_, *tf2_buffer_, target_frame_, 10, this->get_node_logging_interface(),
          this->get_node_clock_interface(), std::chrono::duration<int>(1));
      // Register a callback with tf2_ros::MessageFilter to be called when transforms are available
      tf2_filter_->registerCallback(&EnergyTrackerNode::LeafsCallback, this);
      // Publisher
      target_pub_ = this->create_publisher<auto_aim_interfaces::msg::EnTarget>(
          "/tracker/EnTarget", rclcpp::SensorDataQoS());
   }
   void EnergyTrackerNode::LeafsCallback(const auto_aim_interfaces::msg::Leafs::SharedPtr leafs_msg)
   {
      if (leafs_msg->leafs.empty())
         return;

      // find the best match leaf
      auto leaf_ = leafs_msg->leafs[0];
      for (auto &leaf : leafs_msg->leafs)
      {
         if (leaf_.prob > leaf.prob)
         {
            leaf_ = leaf;
         }
      }
      geometry_msgs::msg::PoseStamped ps;
      ps.header = leafs_msg->header;
      ps.pose = leaf_.pose;
      try
      {
         leaf_.pose = tf2_buffer_->transform(ps, target_frame_).pose;
      }
      catch (const tf2::ExtrapolationException &ex)
      {
         RCLCPP_ERROR(get_logger(), "Error while transforming %s", ex.what());
         return;
      }

      // Init message
      // auto_aim_interfaces::msg::TrackerInfo info_msg;
      auto_aim_interfaces::msg::EnTarget target_msg;
      rclcpp::Time time = leafs_msg->header.stamp;
      target_msg.header.stamp = time;
      target_msg.header.frame_id = target_frame_;

      // Update tracker
      tracker_->init(leaf_);
      t_ = (time - last_time_).seconds();
      tracker_->update(leaf_);
      const auto &state = tracker_->target_state;
      double angle_ = state(0);
      Eigen::Vector2d p1(leaf_.leaf_center.z, leaf_.leaf_center.y);
      Eigen::Vector2d p2(leaf_.r_center.z, leaf_.r_center.y);
      double r_distance = (p1 - p2).norm();
      target_msg.position.y = r_distance * cos(angle_);
      target_msg.position.z = r_distance * sin(angle_);
      target_msg.angle = angle_;
      last_time_ = time;
      target_pub_->publish(target_msg);
   }
} // namespace rm_auto_aim
#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::EnergyTrackerNode)