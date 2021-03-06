#include "layer.hpp"

using namespace cv;

ReLULayer::ReLULayer(Ptr<dnn::Layer> cvLayer) : Layer(cvLayer) {
  auto relu = cvLayer.dynamicCast<dnn::ReLULayer>();
}

std::shared_ptr<ngraph::Node> ReLULayer::initNGraph(std::vector<std::shared_ptr<ngraph::Node> > inputs) {
  return std::make_shared<ngraph::op::Relu>(inputs[0]);
}
