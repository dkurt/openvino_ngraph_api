#include "layer.hpp"

using namespace cv;

SoftMaxLayer::SoftMaxLayer(Ptr<dnn::Layer> cvLayer) : Layer(cvLayer) {}

std::shared_ptr<ngraph::Node> SoftMaxLayer::initNGraph(std::vector<std::shared_ptr<ngraph::Node> > inputs) {
  const int axis = inputs[0]->get_shape().size() - 1;  // axis: -1
  return std::make_shared<ngraph::op::v1::Softmax>(inputs[0], axis);
}
