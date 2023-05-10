#include "layer.hpp"

using namespace cv;

LRNLayer::LRNLayer(Ptr<dnn::Layer> cvLayer) : Layer(cvLayer) {
  auto lrn = cvLayer.dynamicCast<dnn::LRNLayer>();

  alpha = lrn->alpha;
  beta = lrn->beta;
  bias = lrn->bias;
  normBySize = lrn->normBySize;
  size = lrn->size;
  normType = lrn->type == 0 ? "ACROSS_CHANNELS" : "WITHIN_CHANNEL";
}

std::shared_ptr<ngraph::Node> LRNLayer::initNGraph(std::vector<std::shared_ptr<ngraph::Node> > inputs) {
  double alphaSize = alpha;
  if (!normBySize)
      alphaSize *= (normType == "WITHIN_CHANNEL" ? size*size : size);

  return std::make_shared<ngraph::op::LRN>(inputs[0], alpha, beta, bias, size);
}
