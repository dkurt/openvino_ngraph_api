#include "layer.hpp"

using namespace cv;

SoftMaxLayer::SoftMaxLayer(Ptr<dnn::Layer> cvLayer) : Layer(cvLayer) {}

std::shared_ptr<ngraph::Node> SoftMaxLayer::initNGraph(std::shared_ptr<ngraph::Node> input) {
  const int axis = 1;
  return std::make_shared<ngraph::op::v1::Softmax>(input, axis);
}
