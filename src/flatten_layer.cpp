#include "layer.hpp"

using namespace cv;

FlattenLayer::FlattenLayer(Ptr<dnn::Layer> cvLayer) : Layer(cvLayer) {}

std::shared_ptr<ngraph::Node> FlattenLayer::initNGraph(std::vector<std::shared_ptr<ngraph::Node> > inputs) {
  size_t batch = inputs[0]->get_shape()[0];
  size_t spatial = 1;
  for (int i = 1; i < inputs[0]->get_shape().size(); ++i)
      spatial *= inputs[0]->get_shape()[i];

  std::vector<size_t> data = {batch, spatial};
  auto shape = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{2}, data.data());
  return std::make_shared<ngraph::op::v1::Reshape>(inputs[0], shape, true);
}
