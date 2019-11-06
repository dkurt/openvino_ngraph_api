#include "layer.hpp"

using namespace cv;

PoolingLayer::PoolingLayer(Ptr<dnn::Layer> cvLayer) : Layer(cvLayer) {
  auto pool = cvLayer.dynamicCast<dnn::PoolingLayer>();

  if (pool->type != 0)
      CV_Error(Error::StsNotImplemented, "Not max pooling type");

  kernel_size = pool->kernel_size;
  strides = pool->strides;
  pads_begin = pool->pads_begin;
  pads_end = pool->pads_end;
  padMode = pool->padMode;
  ceilMode = pool->ceilMode;
}

std::shared_ptr<ngraph::Node> PoolingLayer::initNGraph(std::shared_ptr<ngraph::Node> input) {
  auto rounding_type = ceilMode ? ngraph::op::RoundingType::CEIL : ngraph::op::RoundingType::FLOOR;

  auto pad_type = ngraph::op::PadType::EXPLICIT;
  if (!padMode.empty()) {
    if (padMode == "VALID")
      pad_type = ngraph::op::PadType::VALID;
    else if (padMode == "SAME")
      pad_type = ngraph::op::PadType::SAME_UPPER;
    else
      CV_Error(Error::StsNotImplemented, "Unknown pad mode: " + padMode);
  }

  return std::make_shared<ngraph::op::v1::MaxPool>(input, ngraph::Strides(strides),
             ngraph::Shape(pads_begin), ngraph::Shape(pads_end), ngraph::Shape(kernel_size),
             rounding_type, pad_type);
}
