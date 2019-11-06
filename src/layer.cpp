#include "layer.hpp"

Layer::Layer(Ptr<dnn::Layer> cvLayer) : name(cvLayer->name) {}

std::shared_ptr<ngraph::op::Constant> wrapMatToConstant(const Mat& m, const std::vector<size_t>& shape) {
  CV_Assert(m.type() == CV_32F);

  std::vector<size_t> constShape;
  if (shape.empty()) {
    constShape = std::vector<size_t>(&m.size[0], &m.size[0] + m.dims);
  } else {
    constShape = shape;
  }
  return std::make_shared<ngraph::op::Constant>(ngraph::element::f32, constShape, m.data);
}
