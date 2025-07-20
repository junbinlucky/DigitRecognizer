#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <numeric>

constexpr int32_t input_c = 1;
constexpr int32_t input_h = 28;
constexpr int32_t input_w = 28;

template <typename T, size_t N>
size_t argmax_after_softmax(const std::array<T, N> &input)
{
    static_assert(std::is_floating_point_v<T>,
                  "Input array must contain floating-point values");

    if constexpr (N == 0)
        return 0;

    // 数值稳定处理
    const T max_val = *std::max_element(input.begin(), input.end());
    T sum_exp = 0;
    std::array<T, N> softmax;

    // 计算指数和
    std::transform(input.begin(), input.end(), softmax.begin(),
                   [max_val](T x)
                   { return std::exp(x - max_val); });
    // 查找最大值索引
    return std::distance(
        softmax.begin(),
        std::max_element(softmax.begin(), softmax.end()));
}

int main()
{
    // Init run envionment.
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MNIST_ENV");
    Ort::SessionOptions session_options;

    // CPU exclusive configuration.
    session_options.SetIntraOpNumThreads(4);                     // Set the number of parallel threads.
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC); // Start basic graph optimization.

    // Load model.
    Ort::Session session(env, "model.onnx", session_options);

    // Prepare the input image data.
    cv::Mat image = cv::imread("mnist/MNIST/raw/img/pic16.png", cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
        std::cerr << "Image read failed." << std::endl;
        return false;
    }
    // Check the image size 28 * 28
    if (image.rows != input_h || image.cols != input_w)
    {
        cv::resize(image, image, cv::Size(28, 28));
    }
    // Normalize image pix value to [0, 1]
    cv::Mat normalized_image;
    image.convertTo(normalized_image, CV_32F, 1. / 255.);

    std::vector<float> input_image = normalized_image.reshape(1, 1);
    std::array<float, 10> results = {};
    std::array<int64_t, 4> input_shape = {1, 1, input_w, input_h};
    std::array<int64_t, 2> output_shape = {1, 10};

    // Create input tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_image.data(), input_image.size(),
                                                              input_shape.data(), input_shape.size());
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(),
                                                               output_shape.data(), output_shape.size());
    const char *input_names[] = {"input.1"};
    const char *output_names[] = {"22"};
    Ort::RunOptions run_options;
    session.Run(run_options, input_names, &input_tensor, 1, output_names, &output_tensor, 1);

    // Print the results.
    std::cout << "Every value confidence is:";
    for (const float value : results)
    {
        std::cout << " " << value;
    }
    std::cout << std::endl;

    const int out = argmax_after_softmax(results);
    std::cout << "Value is: " << out << std::endl;
    // Release the resource.
    session_options.release();
    session.release();
    return 0;
}
