#include "layer.hpp"

using namespace cv;

PoolingLayer::PoolingLayer(Ptr<dnn::Layer> cvLayer) : Layer(cvLayer) {
  auto pool = cvLayer.dynamicCast<dnn::PoolingLayer>();

  if (pool->type == 0)
    type = "max";
  else if (pool->type == 1)
    type = "ave";
  else
      CV_Error(Error::StsNotImplemented, "Supported only MAX and AVE poolings.");

  kernel_size = pool->kernel_size;
  strides = pool->strides;
  pads_begin = pool->pads_begin;
  pads_end = pool->pads_end;
  padMode = pool->padMode;
  ceilMode = pool->ceilMode;
  globalPooling = pool->globalPooling;
}

std::shared_ptr<ngraph::Node> PoolingLayer::initNGraph(std::vector<std::shared_ptr<ngraph::Node> > inputs) {
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

  if (globalPooling) {
    kernel_size.resize(2);
    kernel_size[0] = inputs[0]->get_shape()[2];
    kernel_size[1] = inputs[0]->get_shape()[3];
  }

  if (type == "max") {
    return std::make_shared<ngraph::op::v1::MaxPool>(inputs[0], ngraph::Strides(strides),
               ngraph::Shape(pads_begin), ngraph::Shape(pads_end), ngraph::Shape(kernel_size),
               rounding_type, pad_type);
  } else if (type == "ave") {
    return std::make_shared<ngraph::op::v1::AvgPool>(inputs[0], ngraph::Strides(strides),
               ngraph::Shape(pads_begin), ngraph::Shape(pads_end), ngraph::Shape(kernel_size),
               false /*exclude_pad*/, rounding_type, pad_type);
  }
}
