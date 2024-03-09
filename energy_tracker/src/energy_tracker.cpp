#include "energy_tracker/energy_tracker.hpp"
namespace rm_auto_aim
{
    EnTarcker::EnTarcker()
    {
    }

    double EnTarcker::angleSolver(auto_aim_interfaces::msg::Leaf leaf)
    {
        double angle = atan2(leaf.leaf_center.y - leaf.r_center.y, leaf.leaf_center.x - leaf.r_center.x);
        return angle;
    }

    void EnTarcker::init(const Leaf &l)
    {
        initEKF(l);
        RCLCPP_DEBUG(rclcpp::get_logger("energy_tracker"), "Init EKF!");
    }

    void EnTarcker::update(const Leaf &l)
    {
        Eigen::VectorXd ekf_prediction = ekf.predict();
        RCLCPP_DEBUG(rclcpp::get_logger("energy_tracker"), "EKF predict");

        target_state = ekf_prediction;
        measurement = Eigen::VectorXd();
        measurement << angleSolver(l);
        target_state = ekf.update(measurement);
        RCLCPP_DEBUG(rclcpp::get_logger("energy_tracker"), "EKF update");
    }

    void EnTarcker::initEKF(const Leaf &l)
    {
        target_state = Eigen::VectorXd::Zero(3);
        double angle = angleSolver(l);
        target_state << angle, 0, 0;
        ekf.setState(target_state);
    }
}
