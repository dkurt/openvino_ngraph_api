#include "layer.hpp"

using namespace cv;

PermuteLayer::PermuteLayer(Ptr<dnn::Layer> cvLayer) : Layer(cvLayer) {}

std::shared_ptr<ngraph::Node> PermuteLayer::initNGraph(std::vector<std::shared_ptr<ngraph::Node> > inputs) {
  std::vector<size_t> order{0, 2, 3, 1};
  auto tr_axes = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                     ngraph::Shape({order.size()}), order.data());
  return std::make_shared<ngraph::op::Transpose>(inputs[0], tr_axes);
}
