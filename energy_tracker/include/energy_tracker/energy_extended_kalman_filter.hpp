#ifndef ENERGY__KALMAN_FILTER_HPP_
#define ENERGY__KALMAN_FILTER_HPP_

#include <Eigen/Dense>
#include <functional>

namespace rm_auto_aim
{

class ExtendedKalmanFilter
{
public:
  ExtendedKalmanFilter();
  using VecVecFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd &)>;
  using VecMatFunc = std::function<Eigen::MatrixXd(const Eigen::VectorXd &)>;
  using VoidMatFunc = std::function<Eigen::MatrixXd()>;

  explicit ExtendedKalmanFilter(
    const VecVecFunc & f, const VecVecFunc & h, const VecMatFunc & j_f, const VecMatFunc & j_h,
    const VoidMatFunc & u_q, const VecMatFunc & u_r, const Eigen::MatrixXd & P0);
  // Set the initial state
  void setState(const Eigen::VectorXd & x0);
  // Compute a predicted state
  Eigen::MatrixXd predict();

  // Update the estimated state based on measurement
  Eigen::MatrixXd update(const Eigen::VectorXd & z);
  
private:
  double t=1.0;  //  predict time
  // Posteriori state
  Eigen::VectorXd x_post; // State vector
  // Priori state
  Eigen::VectorXd x_pri;
  Eigen::MatrixXd F; // State transition matrix
  Eigen::MatrixXd P; // Estimate error covariance
  Eigen::MatrixXd Q; // Process noise covariance
  Eigen::MatrixXd H; // Measurement matrix
  Eigen::MatrixXd R; // Measurement noise covariance
};

}  // namespace rm_auto_aim

#endif  // ENERGY__KALMAN_FILTER_HPP_