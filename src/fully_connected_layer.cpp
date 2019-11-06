#include "layer.hpp"

using namespace cv;

FullyConnectedLayer::FullyConnectedLayer(Ptr<dnn::Layer> cvLayer) : Layer(cvLayer) {
  auto fc = cvLayer.dynamicCast<dnn::InnerProductLayer>();

  weights = fc->blobs[0];
  if (fc->blobs.size() > 1)
      biases = fc->blobs[1];
}

std::shared_ptr<ngraph::Node> FullyConnectedLayer::initNGraph(std::vector<std::shared_ptr<ngraph::Node> > inputs) {

  // Do reshape to 2D tensor
  int batch = inputs[0]->get_shape()[0];
  std::vector<size_t> data = {(size_t)batch, (size_t)weights.size[1]};
  auto new_shape = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{2}, data.data());
  auto reshapedInp = std::make_shared<ngraph::op::v1::Reshape>(inputs[0], new_shape, true);

  auto ieWeights = wrapMatToConstant(weights);
  auto matmul = std::make_shared<ngraph::op::MatMul>(reshapedInp, ieWeights, false, true);

  if (!biases.empty()) {
    auto ieBiases = wrapMatToConstant(biases, {(size_t)biases.size[1]});
    return std::make_shared<ngraph::op::Add>(matmul, ieBiases, ngraph::op::AutoBroadcastType::NUMPY);
  }

  return matmul;
}
