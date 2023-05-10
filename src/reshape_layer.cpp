#include "layer.hpp"

using namespace cv;

ReshapeLayer::ReshapeLayer(Ptr<dnn::Layer> cvLayer) : Layer(cvLayer) {
  std::vector<int> _shape = cvLayer.dynamicCast<dnn::ReshapeLayer>()->newShapeDesc;
  shape = std::vector<size_t>(_shape.begin(), _shape.end());
}

std::shared_ptr<ngraph::Node> ReshapeLayer::initNGraph(std::vector<std::shared_ptr<ngraph::Node> > inputs) {
  auto shapeNode = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{shape.size()}, shape.data());
  return std::make_shared<ngraph::op::v1::Reshape>(inputs[0], shapeNode, true);
}
