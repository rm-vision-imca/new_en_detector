// ROS
#include <message_filters/subscriber.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_listener.h>

#include <rclcpp/rclcpp.hpp>
//#include <std_srvs/srv/trigger.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
//#include <visualization_msgs/msg/marker_array.hpp>

// STD
#include <memory>
#include <string>
#include <vector>

#include "energy_extended_kalman_filter.hpp"
#include "auto_aim_interfaces/msg/leafs.hpp"
#include "auto_aim_interfaces/msg/en_target.hpp"
#include "auto_aim_interfaces/msg/tracker_info.hpp"
#include "energy_tracker.hpp"

namespace rm_auto_aim
{
  using tf2_filter = tf2_ros::MessageFilter<auto_aim_interfaces::msg::Leafs>;
  class EnergyTrackerNode : public rclcpp::Node
  {
  public:
    explicit EnergyTrackerNode(const rclcpp::NodeOptions &options);
    
  private:
    void LeafsCallback(const auto_aim_interfaces::msg::Leafs::SharedPtr leafs_msg);

    // The time when the last message was received
    rclcpp::Time last_time_;
    double t_;
    // Armor tracker
    std::unique_ptr<EnTarcker> tracker_;
    // Subscriber with tf2 message_filter
    std::string target_frame_;
    std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;
    message_filters::Subscriber<auto_aim_interfaces::msg::Leafs> leafs_sub_;
    std::shared_ptr<tf2_filter> tf2_filter_;
    rclcpp::Publisher<auto_aim_interfaces::msg::EnTarget>::SharedPtr target_pub_;
  };
}