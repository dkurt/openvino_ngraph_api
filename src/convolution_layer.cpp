#include "layer.hpp"

using namespace cv;

ConvolutionLayer::ConvolutionLayer(Ptr<dnn::Layer> cvLayer) : Layer(cvLayer) {
  auto conv = cvLayer.dynamicCast<dnn::ConvolutionLayer>();

  weights = conv->blobs[0];
  if (conv->blobs.size() > 1)
      biases = conv->blobs[1];
  strides = conv->strides;
  dilations = conv->dilations;
  pads_begin = std::vector<std::ptrdiff_t>(conv->pads_begin.begin(),
                                           conv->pads_begin.end());
  pads_end = std::vector<std::ptrdiff_t>(conv->pads_end.begin(),
                                         conv->pads_end.end());
  padMode = conv->padMode;
}

std::shared_ptr<ngraph::Node> ConvolutionLayer::initNGraph(std::vector<std::shared_ptr<ngraph::Node> > inputs) {
  const int outChannels = weights.size[0];
  const int inpChannels = inputs[0]->get_shape()[1];
  int group = inpChannels / weights.size[1];

  std::vector<size_t> kernel_shape(&weights.size[0], &weights.size[0] + 4);
  if (group != 1)
  {
    kernel_shape[0] /= group;
    kernel_shape.insert(kernel_shape.begin(), group);
  }

  auto ieWeights = wrapMatToConstant(weights, kernel_shape);

  auto pad_type = ngraph::op::PadType::EXPLICIT;
  if (!padMode.empty()) {
    if (padMode == "VALID")
      pad_type = ngraph::op::PadType::VALID;
    else if (padMode == "SAME")
      pad_type = ngraph::op::PadType::SAME_UPPER;
    else
      CV_Error(Error::StsNotImplemented, "Unknown pad mode: " + padMode);
  }

  std::shared_ptr<ngraph::Node> conv;
  if (group == 1) {
    conv = std::make_shared<ngraph::op::v1::Convolution>(
                  inputs[0], ieWeights,
                  ngraph::Strides(strides),
                  ngraph::CoordinateDiff(pads_begin),
                  ngraph::CoordinateDiff(pads_end),
                  ngraph::Strides(dilations),
                  pad_type);
  }
  else {
    conv = std::make_shared<ngraph::op::v1::GroupConvolution>(
                  inputs[0], ieWeights,
                  ngraph::Strides(strides),
                  ngraph::CoordinateDiff(pads_begin),
                  ngraph::CoordinateDiff(pads_end),
                  ngraph::Strides(dilations),
                  pad_type);
  }

  if (!biases.empty())
  {
    std::vector<size_t> shape(conv->get_shape().size(), 1);
    shape[1] = outChannels;
    auto ieBiases = wrapMatToConstant(biases, shape);
    return std::make_shared<ngraph::op::Add>(conv, ieBiases, ngraph::op::AutoBroadcastType::NUMPY);
  }

  return conv;
}
