#include "layer.hpp"

using namespace cv;

ConcatLayer::ConcatLayer(Ptr<dnn::Layer> cvLayer) : Layer(cvLayer) {}

std::shared_ptr<ngraph::Node> ConcatLayer::initNGraph(std::vector<std::shared_ptr<ngraph::Node> > inputs) {
  const int axis = 1;
  return std::make_shared<ngraph::op::Concat>(inputs, axis);
}
