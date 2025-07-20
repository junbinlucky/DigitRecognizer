#include <glog/logging.h>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>
#include "data/tensor.hpp"
#include "runtime/runtime_ir.hpp"

using namespace kuiper_infer;

constexpr int32_t input_c = 1;
constexpr int32_t input_h = 28;
constexpr int32_t input_w = 28;

bool MnistDemo(const std::string& image_path, cv::Mat& normalized_image) {
  cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
  if (image.empty()) {
    std::cerr << "Image read failed." << std::endl;
    return false;
  }
  // Check the image size 28 * 28
  if (image.rows != input_h || image.cols != input_w) {
    cv::resize(image, image, cv::Size(28, 28));
  }
  // Normalize image pix value to [0, 1]
  image.convertTo(normalized_image, CV_32F, 1. / 255.);
  return true;
}

int main() {
  const uint32_t batch_size = 1;
  const std::string& image_path = "./imgs/mnist/pic19.png"; 
  const std::string& param_path = "tmp/mnist/demo/model_torchscript.pnnx.param";
  const std::string& bin_path = "tmp/mnist/demo/model_torchscript.pnnx.bin";

  cv::Mat normalized_image;
  if (!MnistDemo(image_path, normalized_image)) {
    return 1;
  }
  sftensor input = std::make_shared<ftensor>(input_c, input_h, input_w);
  input->Fill(0.f);
  const cv::Mat image_t = normalized_image.t();
  memcpy(input->slice(0).memptr(), image_t.data, sizeof(float) * normalized_image.total());

  std::vector<sftensor> inputs;
  inputs.push_back(input);

  RuntimeGraph graph(param_path, bin_path);
  graph.Build();
  graph.set_inputs("pnnx_input_0", inputs);
  graph.Forward(true);
  const std::vector<sftensor> outputs = graph.get_outputs("pnnx_output_0");
  assert(outputs.size() == inputs.size());
  assert(outputs.size() == batch_size);
  for (uint32_t i = 0; i < batch_size; ++i) {
    const sftensor& output = outputs.at(i);
    assert(!output->empty());
    const auto& shapes = output->shapes();
    std::cerr << "Shape size is: " << shapes.size() << std::endl;
    assert(shapes.size() == 3);
    const uint32_t num_info = shapes.at(2);
    std::cout << "Every value confidence is:";
    int best_id = -1;
    int best_confidence = -1.f;
    for (uint32_t j = 0; j < num_info; ++j) {
      const float confidence = output->at(0, 0, j);
      if (confidence > best_confidence) {
        best_confidence = confidence;
        best_id = j;
      }
      std::cout << " " << confidence;
    }
    std::cout << std::endl;
    std::cout << "Number is: " << best_id << std::endl;
  }

  return 0;
}