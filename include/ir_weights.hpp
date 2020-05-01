#include <queue>

#include <opencv2/opencv.hpp>

// This class is used to extract weights from OpenVINO IR
class IRWeights {
public:
  IRWeights(const std::string& xmlPath, const std::string& binPath);

  cv::Mat next();

private:
  std::queue<cv::Mat> weights;
};
