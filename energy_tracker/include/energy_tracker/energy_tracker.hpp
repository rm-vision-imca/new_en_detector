#ifndef ENERGY_TRACKER_TRACKER_HPP_
#define ENERGY_TRACKER_TRACKER_HPP_

// Eigen
#include <Eigen/Eigen>

// ROS
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <rclcpp/logger.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// STD
#include <memory>
#include <string>

#include "energy_extended_kalman_filter.hpp"
#include "auto_aim_interfaces/msg/leaf.hpp"
#include "auto_aim_interfaces/msg/leafs.hpp"

namespace rm_auto_aim
{
    class EnTarcker
    {
        public:
        EnTarcker();
        double angleSolver(auto_aim_interfaces::msg::Leaf leaf);
        using Leafs = auto_aim_interfaces::msg::Leafs;
        using Leaf = auto_aim_interfaces::msg::Leaf;
        void init(const Leaf & leafs_msg);
        void update(const Leaf & leafs_msg);
        Eigen::VectorXd target_state;
        ExtendedKalmanFilter ekf;
        Eigen::VectorXd measurement;
        private:
        void initEKF(const Leaf & a);
        Leaf tracked_leaf;

    };


}
#endif