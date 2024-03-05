#ifndef DETECTOR_HPP_
#define DETECTOR_HPP_

#include <sys/types.h>

#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <inference_engine.hpp>
#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "energy_detector/Leaf.hpp"
#include "auto_aim_interfaces/msg/debug_leafs.hpp"
using namespace InferenceEngine;


#define IMG_SIZE 640         // 推理图像大小
#define DEVICE "CPU"         // 设备选择
#define KPT_NUM 4
#define CLS_NUM 1
#ifdef CURRENT_PKG_DIR
#define MODEL_PATH CURRENT_PKG_DIR "/models/best_fp16.xml"
#endif

namespace rm_auto_aim
{
  class Detector
  {
  public:
    Detector(float NMS_THRESHOLD,float CONF_THRESHOLD,int color);
    std::vector<Leaf> detect(const cv::Mat &input);
    cv::Mat letter_box(cv::Mat &src);
    std::vector<Leaf> work(cv::Mat src_img);
    std::vector<Leaf> Leaf_filter(
        std::vector<Leaf> &leafs, const int MAX_WIDTH, const int MAX_HEIGHT);
    void drawRuselt(cv::Mat &src);
    cv::Point2f fingR(cv::Mat src);
    // Debug msgs
    cv::Mat result_img;
    auto_aim_interfaces::msg::DebugLeafs debug_leafs;
    std::vector<Leaf> leafs_;
    //params
    float NMS_THRESHOLD; // NMS参数
    float CONF_THRESHOLD; // 置信度参数
    int detect_color;

  private:
    cv::Mat dilate_struct;
    cv::Mat erode_struct;
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    ov::Output<const ov::Node>input_port;
    const std::vector<std::string> class_names = {"leaf"};

  };
} // namespace rm_auto_aim
#endif // DETECTOR_HPP_
