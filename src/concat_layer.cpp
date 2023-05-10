#include "layer.hpp"

using namespace cv;

ConcatLayer::ConcatLayer(Ptr<dnn::Layer> cvLayer) : Layer(cvLayer) {
  axis = cvLayer.dynamicCast<dnn::ConcatLayer>()->axis;
}

std::shared_ptr<ngraph::Node> ConcatLayer::initNGraph(std::vector<std::shared_ptr<ngraph::Node> > inputs) {
  return std::make_shared<ngraph::op::Concat>(inputs, axis);
}
